CFLAGS = -O3 -qstrict -lm

All: prog_cpu prog_omp prog_mpi prog_combo

prog_cpu: cpu/main_cpu.c
	xlc $(CFLAGS) $< -o $@;

prog_omp: omp/main_omp.c
	xlc $(CFLAGS) -qsmp=omp $< -o $@;

prog_mpi: mpi/main_mpi.c
	module load SpectrumMPI && mpixlc $(CFLAGS) $< -o $@;

prog_combo: combo/main_combo.c
	module load SpectrumMPI && mpixlc $(CFLAGS) -qsmp=omp $< -o $@;

clean:
	rm -rf prog_cpu prog_omp prog_mpi prog_combo;
