CC = xlc
CFLAGS = -O3 -qstrict -lm

All: prog_cpu prog_omp

prog_cpu: main_cpu.c
	$(CC) $(CFLAGS) $< -o $@

prog_omp: main_omp.c
	$(CC) $(CFLAGS) -qsmp=omp $< -o $@

clean:
	rm -rf prog_cpu prog_omp

