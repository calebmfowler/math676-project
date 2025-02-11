/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 1999 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

void first_grid() {
  Triangulation<2> triangulation;

  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(4);

  std::ofstream out("grid-1.svg");
  GridOut       grid_out;
  grid_out.write_svg(triangulation, out);
  std::cout << "Grid written to grid-1.svg" << std::endl;
}

void second_grid() {
  Triangulation<2> triangulation;
  const Point<2> center(1, 0);
  const double   inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell(
    triangulation, center, inner_radius, outer_radius, 10);
  for (unsigned int step = 0; step < 5; ++step) {
    for (const auto &cell : triangulation.active_cell_iterators()) {
      for (const auto v : cell->vertex_indices()) {
        const double distance_from_center = center.distance(cell->vertex(v));
        if (std::fabs(distance_from_center - inner_radius) <= 1e-6 * inner_radius) {
          cell->set_refine_flag();
          break;
        }
      }
    }
    triangulation.execute_coarsening_and_refinement();
  }
  std::ofstream out("grid-2.svg");
  GridOut       grid_out;
  grid_out.write_svg(triangulation, out);
  std::cout << "Grid written to grid-2.svg" << std::endl;
}

void third_grid() {
  Triangulation<2> triangulation;
  const Point<2> center(1, 0);
  const double   inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius, 10);
  for (unsigned int step = 0; step < 5; ++step) {
    for (const auto &cell : triangulation.active_cell_iterators()) {
      for (const auto v : cell->vertex_indices()) {
          const double distance_from_center = center.distance(cell->vertex(v));
          if (distance_from_center >= 0.75 - 1e-6 * inner_radius && distance_from_center <= 1.0 + 1e-6 * inner_radius) {
            cell->set_refine_flag();
            break;
          }
        }
      }
    triangulation.execute_coarsening_and_refinement();
  }
  std::ofstream out("grid-3.svg");
  GridOut       grid_out;
  grid_out.write_svg(triangulation, out);
  std::cout << "Grid written to grid-3.svg" << std::endl;
}

int main() {
  first_grid();
  second_grid();
  third_grid();
}