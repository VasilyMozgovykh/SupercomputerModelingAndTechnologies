CFLAGS = -O3 -qstrict -lm

All: prog_cpu prog_omp prog_mpi

prog_cpu: main_cpu.c
        xlc $(CFLAGS) $< -o $@;

prog_omp: main_omp.c
        xlc $(CFLAGS) -qsmp=omp $< -o $@;

prog_mpi: main_mpi.c
        mpixlc $(CFLAGS) $< -o $@;

clean:
        rm -rf prog_cpu prog_omp prog_mpi;