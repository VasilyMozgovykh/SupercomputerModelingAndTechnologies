make prog_mpi;
mkdir bsub_results;
mkdir bsub_results/mpi;
PREFIX="bsub_results/mpi";
mpisubmit.pl -p 2 -t 1 -stdout "$PREFIX/fout_600x400_2.txt" ./prog_mpi 600 400;
mpisubmit.pl -p 4 -t 1 -stdout "$PREFIX/fout_600x400_4.txt" ./prog_mpi 600 400;
mpisubmit.pl -p 8 -t 1 -stdout "$PREFIX/fout_600x400_8.txt" ./prog_mpi 600 400;
mpisubmit.pl -p 16 -t 1 -stdout "$PREFIX/fout_600x400_16.txt" ./prog_mpi 600 400;
