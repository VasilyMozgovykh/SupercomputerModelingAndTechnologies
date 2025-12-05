make prog_mpi;
mkdir bsub_results;
mkdir bsub_results/mpi;
PREFIX="bsub_results/mpi";
mpisubmit.pl -p 4 -t 1 -stdout "$PREFIX/fout_1200x800_4.txt" -stderr "$PREFIX/ferr_1200x800_4.txt" ./prog_mpi 1200 800;
mpisubmit.pl -p 8 -t 1 -stdout "$PREFIX/fout_1200x800_8.txt" -stderr "$PREFIX/ferr_1200x800_8.txt" ./prog_mpi 1200 800;
mpisubmit.pl -p 16 -t 1 -stdout "$PREFIX/fout_1200x800_16.txt" -stderr "$PREFIX/ferr_1200x800_16.txt" ./prog_mpi 1200 800;
mpisubmit.pl -p 32 -t 1 -stdout "$PREFIX/fout_1200x800_32.txt" -stderr "$PREFIX/ferr_1200x800_32.txt" ./prog_mpi 1200 800;
