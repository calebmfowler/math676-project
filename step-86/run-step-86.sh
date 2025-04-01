cmake .
mpirun -np 4 ./step-86 heat_equation.prm
rm -r "/mnt/c/Users/Caleb Fowler/Documents/MATH 676/step-86-solution"
mkdir "/mnt/c/Users/Caleb Fowler/Documents/MATH 676/step-86-solution"
mv -f solution* "/mnt/c/Users/Caleb Fowler/Documents/MATH 676/step-86-solution"