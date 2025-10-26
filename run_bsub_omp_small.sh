make prog_omp;
mkdir bsub_results;
mkdir bsub_results/omp;
PREFIX="bsub_results/omp";
bsub -n 1 -W 1 -oo "$PREFIX/fout_600x400_2.txt" -eo "$PREFIX/ferr_600x400_2.txt" ./prog_omp 600 400 2;
bsub -n 1 -W 1 -oo "$PREFIX/fout_600x400_4.txt" -eo "$PREFIX/ferr_600x400_4.txt" ./prog_omp 600 400 4;
bsub -n 1 -W 1 -oo "$PREFIX/fout_600x400_8.txt" -eo "$PREFIX/ferr_600x400_8.txt" ./prog_omp 600 400 8;
bsub -n 1 -W 1 -oo "$PREFIX/fout_600x400_16.txt" -eo "$PREFIX/ferr_600x400_16.txt" ./prog_omp 600 400 16;
bsub -n 1 -W 1 -oo "$PREFIX/fout_600x400_32.txt" -eo "$PREFIX/ferr_600x400_32.txt" ./prog_omp 600 400 32;

