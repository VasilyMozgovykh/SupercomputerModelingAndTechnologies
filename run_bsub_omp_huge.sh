make prog_omp;
mkdir bsub_results;
mkdir bsub_results/omp;
PREFIX="bsub_results/omp";
bsub -n 1 -W 1 -oo "$PREFIX/fout_1200x800_2.txt" -eo "$PREFIX/ferr_1200x800_2.txt" ./prog_omp 1200 800 2;
bsub -n 1 -W 1 -oo "$PREFIX/fout_1200x800_4.txt" -eo "$PREFIX/ferr_1200x800_4.txt" ./prog_omp 1200 800 4;
bsub -n 1 -W 1 -oo "$PREFIX/fout_1200x800_8.txt" -eo "$PREFIX/ferr_1200x800_8.txt" ./prog_omp 1200 800 8;
bsub -n 1 -W 1 -oo "$PREFIX/fout_1200x800_16.txt" -eo "$PREFIX/ferr_1200x800_16.txt" ./prog_omp 1200 800 16;
bsub -n 1 -W 1 -oo "$PREFIX/fout_1200x800_32.txt" -eo "$PREFIX/ferr_1200x800_32.txt" ./prog_omp 1200 800 32;

