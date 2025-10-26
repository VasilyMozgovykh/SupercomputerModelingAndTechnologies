make prog_cpu;
mkdir bsub_results;
mkdir bsub_results/cpu;
PREFIX="bsub_results/cpu";
bsub -n 1 -W 1 -o "$PREFIX/fout_600x400.txt" -e "$PREFIX/ferr_600x400.txt" ./prog_cpu 600 400;
bsub -n 1 -W 1 -o "$PREFIX/fout_1200x800.txt" -e "$PREFIX/ferr_1200x800.txt" ./prog_cpu 1200 800;

