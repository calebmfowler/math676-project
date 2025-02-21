/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2000 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Authors:
 *   Wolfgang Bangerth, Colorado State University, 2024
 *   Stefano Zampini, King Abdullah University of Science and Technology, 2024
 */


#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/petsc_ts.h>

#include <fstream>
#include <iostream>


namespace Step86
{
  using namespace dealii;

  template <int dim>
  class HeatEquation : public ParameterAcceptor
  {
  public:
    HeatEquation(const MPI_Comm mpi_communicator);
    void run();

  private:
    const MPI_Comm mpi_communicator;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    parallel::distributed::Triangulation<dim> triangulation;
    FE_Q<dim>                                 fe;
    DoFHandler<dim>                           dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;


    void setup_system(const double time);

    void output_results(const double                      time,
                        const unsigned int                timestep_number,
                        const PETScWrappers::MPI::Vector &solution);

    void implicit_function(const double                      time,
                           const PETScWrappers::MPI::Vector &solution,
                           const PETScWrappers::MPI::Vector &solution_dot,
                           PETScWrappers::MPI::Vector       &residual);

    void
    assemble_implicit_jacobian(const double                      time,
                               const PETScWrappers::MPI::Vector &solution,
                               const PETScWrappers::MPI::Vector &solution_dot,
                               const double                      shift);

    void solve_with_jacobian(const PETScWrappers::MPI::Vector &src,
                             PETScWrappers::MPI::Vector       &residual);

    PETScWrappers::MPI::SparseMatrix jacobian_matrix;


    void prepare_for_coarsening_and_refinement(
      const PETScWrappers::MPI::Vector &solution);

    void transfer_solution_vectors_to_new_mesh(
      const double                                   time,
      const std::vector<PETScWrappers::MPI::Vector> &all_in,
      std::vector<PETScWrappers::MPI::Vector>       &all_out);

    AffineConstraints<double> hanging_nodes_constraints;
    AffineConstraints<double> current_constraints;
    AffineConstraints<double> homogeneous_constraints;

    void update_current_constraints(const double time);

    PETScWrappers::TimeStepperData time_stepper_data;

    unsigned int initial_global_refinement;
    unsigned int max_delta_refinement_level;
    unsigned int mesh_adaptation_frequency;

    ParameterAcceptorProxy<Functions::ParsedFunction<dim>>
      right_hand_side_function;
    ParameterAcceptorProxy<Functions::ParsedFunction<dim>>
      initial_value_function;
    ParameterAcceptorProxy<Functions::ParsedFunction<dim>>
      boundary_values_function;
  };


  template <int dim>
  HeatEquation<dim>::HeatEquation(const MPI_Comm mpi_communicator)
    : ParameterAcceptor("/Heat Equation/")
    , mpi_communicator(mpi_communicator)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , fe(1)
    , dof_handler(triangulation)
    , time_stepper_data("",
                        "beuler",
                        /* start time */ 0.0,
                        /* end time */ 1.0,
                        /* initial time step */ 0.025)
    , initial_global_refinement(5)
    , max_delta_refinement_level(2)
    , mesh_adaptation_frequency(0)
    , right_hand_side_function("/Heat Equation/Right hand side")
    , initial_value_function("/Heat Equation/Initial value")
    , boundary_values_function("/Heat Equation/Boundary values")
  {
    enter_subsection("Time stepper");
    {
      enter_my_subsection(this->prm);
      {
        time_stepper_data.add_parameters(this->prm);
      }
      leave_my_subsection(this->prm);
    }
    leave_subsection();

    add_parameter("Initial global refinement",
                  initial_global_refinement,
                  "Number of times the mesh is refined globally before "
                  "starting the time stepping.");
    add_parameter("Maximum delta refinement level",
                  max_delta_refinement_level,
                  "Maximum number of local refinement levels.");
    add_parameter("Mesh adaptation frequency",
                  mesh_adaptation_frequency,
                  "When to adapt the mesh.");
  }



  template <int dim>
  void HeatEquation<dim>::setup_system(const double time)
  {
    TimerOutput::Scope t(computing_timer, "setup system");

    dof_handler.distribute_dofs(fe);
    pcout << std::endl
          << "Number of active cells: " << triangulation.n_active_cells()
          << std::endl
          << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl
          << std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);


    hanging_nodes_constraints.clear();
    hanging_nodes_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_nodes_constraints);
    hanging_nodes_constraints.make_consistent_in_parallel(locally_owned_dofs,
                                                          locally_relevant_dofs,
                                                          mpi_communicator);
    hanging_nodes_constraints.close();


    homogeneous_constraints.clear();
    homogeneous_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    homogeneous_constraints.merge(hanging_nodes_constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             homogeneous_constraints);
    homogeneous_constraints.make_consistent_in_parallel(locally_owned_dofs,
                                                        locally_relevant_dofs,
                                                        mpi_communicator);
    homogeneous_constraints.close();


    update_current_constraints(time);


    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    homogeneous_constraints,
                                    false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);

    jacobian_matrix.reinit(locally_owned_dofs,
                           locally_owned_dofs,
                           dsp,
                           mpi_communicator);
  }


  template <int dim>
  void HeatEquation<dim>::output_results(const double       time,
                                         const unsigned int timestep_number,
                                         const PETScWrappers::MPI::Vector &y)
  {
    TimerOutput::Scope t(computing_timer, "output results");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(y, "U");
    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, mpi_communicator);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        static std::vector<std::pair<double, std::string>> times_and_names;
        times_and_names.emplace_back(time, filename);

        std::ofstream pvd_output("solution.pvd");
        DataOutBase::write_pvd_record(pvd_output, times_and_names);
      }
  }



  template <int dim>
  void
  HeatEquation<dim>::implicit_function(const double                      time,
                                       const PETScWrappers::MPI::Vector &y,
                                       const PETScWrappers::MPI::Vector &y_dot,
                                       PETScWrappers::MPI::Vector &residual)
  {
    TimerOutput::Scope t(computing_timer, "implicit function");

    PETScWrappers::MPI::Vector tmp_solution(locally_owned_dofs,
                                            mpi_communicator);
    PETScWrappers::MPI::Vector tmp_solution_dot(locally_owned_dofs,
                                                mpi_communicator);
    tmp_solution     = y;
    tmp_solution_dot = y_dot;

    update_current_constraints(time);
    current_constraints.distribute(tmp_solution);
    homogeneous_constraints.distribute(tmp_solution_dot);

    PETScWrappers::MPI::Vector locally_relevant_solution(locally_owned_dofs,
                                                         locally_relevant_dofs,
                                                         mpi_communicator);
    PETScWrappers::MPI::Vector locally_relevant_solution_dot(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    locally_relevant_solution     = tmp_solution;
    locally_relevant_solution_dot = tmp_solution_dot;


    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> solution_gradients(n_q_points);
    std::vector<double>         solution_dot_values(n_q_points);

    Vector<double> cell_residual(dofs_per_cell);

    right_hand_side_function.set_time(time);

    residual = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          fe_values.get_function_gradients(locally_relevant_solution,
                                           solution_gradients);
          fe_values.get_function_values(locally_relevant_solution_dot,
                                        solution_dot_values);

          cell->get_dof_indices(local_dof_indices);

          cell_residual = 0;
          for (const unsigned int q : fe_values.quadrature_point_indices())
            for (const unsigned int i : fe_values.dof_indices())
              {
                cell_residual(i) +=
                  (fe_values.shape_value(i, q) *       // [phi_i(x_q) *
                     solution_dot_values[q]            //  u(x_q)
                   +                                   //  +
                   fe_values.shape_grad(i, q) *        //  grad phi_i(x_q) *
                     solution_gradients[q]             //  grad u(x_q)
                   -                                   //  -
                   fe_values.shape_value(i, q) *       //  phi_i(x_q) *
                     right_hand_side_function.value(   //
                       fe_values.quadrature_point(q))) //  f(x_q)]
                  * fe_values.JxW(q);                  // * dx
              }
          current_constraints.distribute_local_to_global(cell_residual,
                                                         local_dof_indices,
                                                         residual);
        }
    residual.compress(VectorOperation::add);

    for (const auto &c : current_constraints.get_lines())
      if (locally_owned_dofs.is_element(c.index))
        {
          if (c.entries.empty()) /* no dependencies -> a Dirichlet node */
            residual[c.index] = y[c.index] - tmp_solution[c.index];
          else /* has dependencies -> a hanging node */
            residual[c.index] = y[c.index];
        }
    residual.compress(VectorOperation::insert);
  }


  template <int dim>
  void HeatEquation<dim>::assemble_implicit_jacobian(
    const double /* time */,
    const PETScWrappers::MPI::Vector & /* y */,
    const PETScWrappers::MPI::Vector & /* y_dot */,
    const double alpha)
  {
    TimerOutput::Scope t(computing_timer, "assemble implicit Jacobian");

    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    jacobian_matrix = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          cell_matrix = 0;
          for (const unsigned int q : fe_values.quadrature_point_indices())
            for (const unsigned int i : fe_values.dof_indices())
              for (const unsigned int j : fe_values.dof_indices())
                {
                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q) *      // grad phi_i(x_q) *
                       fe_values.shape_grad(j, q)      // grad phi_j(x_q)
                     + alpha *                         //
                         fe_values.shape_value(i, q) * // phi_i(x_q) *
                         fe_values.shape_value(j, q)   // phi_j(x_q)
                     ) *
                    fe_values.JxW(q); // * dx
                }
          current_constraints.distribute_local_to_global(cell_matrix,
                                                         local_dof_indices,
                                                         jacobian_matrix);
        }
    jacobian_matrix.compress(VectorOperation::add);

    for (const auto &c : current_constraints.get_lines())
      jacobian_matrix.set(c.index, c.index, 1.0);
    jacobian_matrix.compress(VectorOperation::insert);
  }


  template <int dim>
  void
  HeatEquation<dim>::solve_with_jacobian(const PETScWrappers::MPI::Vector &src,
                                         PETScWrappers::MPI::Vector       &dst)
  {
    TimerOutput::Scope t(computing_timer, "solve with Jacobian");

#if defined(PETSC_HAVE_HYPRE)
    PETScWrappers::PreconditionBoomerAMG preconditioner;
    preconditioner.initialize(jacobian_matrix);
#else
    PETScWrappers::PreconditionSSOR preconditioner;
    preconditioner.initialize(
      jacobian_matrix, PETScWrappers::PreconditionSSOR::AdditionalData(1.0));
#endif

    SolverControl           solver_control(1000, 1e-8 * src.l2_norm());
    PETScWrappers::SolverCG cg(solver_control);
    cg.set_prefix("user_");

    cg.solve(jacobian_matrix, dst, src, preconditioner);

    pcout << "     " << solver_control.last_step() << " linear iterations."
          << std::endl;
  }


  template <int dim>
  void HeatEquation<dim>::prepare_for_coarsening_and_refinement(
    const PETScWrappers::MPI::Vector &y)
  {
    PETScWrappers::MPI::Vector locally_relevant_solution(locally_owned_dofs,
                                                         locally_relevant_dofs,
                                                         mpi_communicator);
    locally_relevant_solution = y;

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                       locally_relevant_solution,
                                       estimated_error_per_cell);

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation, estimated_error_per_cell, 0.6, 0.4);

    const unsigned int max_grid_level =
      initial_global_refinement + max_delta_refinement_level;
    const unsigned int min_grid_level = initial_global_refinement;

    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();
  }


  template <int dim>
  void HeatEquation<dim>::transfer_solution_vectors_to_new_mesh(
    const double                                   time,
    const std::vector<PETScWrappers::MPI::Vector> &all_in,
    std::vector<PETScWrappers::MPI::Vector>       &all_out)
  {
    parallel::distributed::SolutionTransfer<dim, PETScWrappers::MPI::Vector>
      solution_trans(dof_handler);

    std::vector<PETScWrappers::MPI::Vector> all_in_ghosted(all_in.size());
    std::vector<const PETScWrappers::MPI::Vector *> all_in_ghosted_ptr(
      all_in.size());
    std::vector<PETScWrappers::MPI::Vector *> all_out_ptr(all_in.size());
    for (unsigned int i = 0; i < all_in.size(); ++i)
      {
        all_in_ghosted[i].reinit(locally_owned_dofs,
                                 locally_relevant_dofs,
                                 mpi_communicator);
        all_in_ghosted[i]     = all_in[i];
        all_in_ghosted_ptr[i] = &all_in_ghosted[i];
      }

    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(all_in_ghosted_ptr);
    triangulation.execute_coarsening_and_refinement();

    setup_system(time);

    all_out.resize(all_in.size());
    for (unsigned int i = 0; i < all_in.size(); ++i)
      {
        all_out[i].reinit(locally_owned_dofs, mpi_communicator);
        all_out_ptr[i] = &all_out[i];
      }
    solution_trans.interpolate(all_out_ptr);

    for (PETScWrappers::MPI::Vector &v : all_out)
      hanging_nodes_constraints.distribute(v);
  }


  template <int dim>
  void HeatEquation<dim>::update_current_constraints(const double time)
  {
    if (current_constraints.n_constraints() == 0 ||
        time != boundary_values_function.get_time())
      {
        TimerOutput::Scope t(computing_timer, "update current constraints");

        boundary_values_function.set_time(time);
        current_constraints.clear();
        current_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
        current_constraints.merge(hanging_nodes_constraints);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 boundary_values_function,
                                                 current_constraints);
        current_constraints.make_consistent_in_parallel(locally_owned_dofs,
                                                        locally_relevant_dofs,
                                                        mpi_communicator);
        current_constraints.close();
      }
  }


  template <int dim>
  void HeatEquation<dim>::run()
  {
    GridGenerator::hyper_L(triangulation);
    triangulation.refine_global(initial_global_refinement);

    setup_system(/* time */ 0);

    PETScWrappers::TimeStepper<PETScWrappers::MPI::Vector,
                               PETScWrappers::MPI::SparseMatrix>
      petsc_ts(time_stepper_data);

    petsc_ts.set_matrices(jacobian_matrix, jacobian_matrix);


    petsc_ts.implicit_function = [&](const double                      time,
                                     const PETScWrappers::MPI::Vector &y,
                                     const PETScWrappers::MPI::Vector &y_dot,
                                     PETScWrappers::MPI::Vector       &res) {
      this->implicit_function(time, y, y_dot, res);
    };

    petsc_ts.setup_jacobian = [&](const double                      time,
                                  const PETScWrappers::MPI::Vector &y,
                                  const PETScWrappers::MPI::Vector &y_dot,
                                  const double                      alpha) {
      this->assemble_implicit_jacobian(time, y, y_dot, alpha);
    };

    petsc_ts.solve_with_jacobian = [&](const PETScWrappers::MPI::Vector &src,
                                       PETScWrappers::MPI::Vector       &dst) {
      this->solve_with_jacobian(src, dst);
    };

    petsc_ts.algebraic_components = [&]() {
      IndexSet algebraic_set(dof_handler.n_dofs());
      algebraic_set.add_indices(DoFTools::extract_boundary_dofs(dof_handler));
      algebraic_set.add_indices(
        DoFTools::extract_hanging_node_dofs(dof_handler));
      return algebraic_set;
    };

    petsc_ts.update_constrained_components =
      [&](const double time, PETScWrappers::MPI::Vector &y) {
        TimerOutput::Scope t(computing_timer, "set algebraic components");
        update_current_constraints(time);
        current_constraints.distribute(y);
      };


    petsc_ts.decide_and_prepare_for_remeshing =
      [&](const double /* time */,
          const unsigned int                step_number,
          const PETScWrappers::MPI::Vector &y) -> bool {
      if (step_number > 0 && this->mesh_adaptation_frequency > 0 &&
          step_number % this->mesh_adaptation_frequency == 0)
        {
          pcout << std::endl << "Adapting the mesh..." << std::endl;
          this->prepare_for_coarsening_and_refinement(y);
          return true;
        }
      else
        return false;
    };

    petsc_ts.transfer_solution_vectors_to_new_mesh =
      [&](const double                                   time,
          const std::vector<PETScWrappers::MPI::Vector> &all_in,
          std::vector<PETScWrappers::MPI::Vector>       &all_out) {
        this->transfer_solution_vectors_to_new_mesh(time, all_in, all_out);
      };

    petsc_ts.monitor = [&](const double                      time,
                           const PETScWrappers::MPI::Vector &y,
                           const unsigned int                step_number) {
      pcout << "Time step " << step_number << " at t=" << time << std::endl;
      this->output_results(time, step_number, y);
    };


    PETScWrappers::MPI::Vector solution(locally_owned_dofs, mpi_communicator);
    VectorTools::interpolate(dof_handler, initial_value_function, solution);

    petsc_ts.solve(solution);
  }
} // namespace Step86


int main(int argc, char **argv)
{
  try
    {
      using namespace Step86;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      HeatEquation<2>                  heat_equation_solver(MPI_COMM_WORLD);

      const std::string input_filename =
        (argc > 1 ? argv[1] : "heat_equation.prm");
      ParameterAcceptor::initialize(input_filename, "heat_equation_used.prm");
      heat_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
