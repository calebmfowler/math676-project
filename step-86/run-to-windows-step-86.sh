cd ~/mt676/math676-project/step-86

if [ "$#" -eq 0 ]
then
    test_files=heat_equation_[0-9]*.prm
else
    test_files="$@"
fi

cmake .
make release
make
for test_file in $test_files
do
    rm -r solution*
    mpirun -np 4 ./step-86 "$test_file" -ts_monitor
    rm -r "/mnt/c/Users/Caleb Fowler/Documents/MATH 676/step-86-tests/$test_file"
    mkdir "/mnt/c/Users/Caleb Fowler/Documents/MATH 676/step-86-tests/$test_file"
    mv -f solution* "/mnt/c/Users/Caleb Fowler/Documents/MATH 676/step-86-tests/$test_file"
done