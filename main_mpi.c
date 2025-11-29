#include <malloc.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define XLEFT (-3.2)
#define XRIGHT (3.2)
#define YLEFT (-0.2)
#define YRIGHT (4.2)
#define TOL (3e-5)

#define UPRELU(x, thr) (((x) < (thr)) ? (thr) : (x))
#define DOWNRELU(x, thr) (((x) > (thr)) ? (thr) : (x))
#define TO_SEGMENT(x, x1, x2) (DOWNRELU(UPRELU(x, x1), x2))
#define SQUARE(x) ((x) * (x))
#define INT_LINEAR(k, b, x1, x2) (0.5 * (k) * (SQUARE(x2) - SQUARE(x1)) + (b) * ((x2) - (x1)))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

typedef struct MPISettings {
    int rank;
    int size;
    int xsize;
    int ysize;
    int xrank;
    int yrank;
} MPISettings;

typedef struct DimSettings {
    int xdim;
    int ydim;
} DimSettings;

typedef struct TaskSettings {
    double **coef_a;
    double **coef_b;
    double **coef_F;
    double **coef_D_inv;
    DimSettings global_dim;
    DimSettings local_dim;
    DimSettings offsets;
    MPISettings mpi;
} TaskSettings;

typedef struct TaskResults {
    double **solution;
    double time;
    int iter_num;
} TaskResults;

void get_best_local_dim(TaskSettings *settings);

double **allocate_2D(DimSettings dim);

void free_2D(double **arr, DimSettings dim);

void init_coeffitients(TaskSettings settings);

void init_preconditioning(TaskSettings settings);

TaskResults solve_poisson(TaskSettings settings);

int main(int argc, char **argv) {

    /* Prepare settings */
    TaskSettings settings;
    settings.global_dim.xdim = atoi(argv[1]);
    settings.global_dim.ydim = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &settings.mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &settings.mpi.size);

    get_best_local_dim(&settings);
    settings.coef_a = allocate_2D(settings.local_dim);
    settings.coef_b = allocate_2D(settings.local_dim);
    settings.coef_F = allocate_2D(settings.local_dim);
    settings.coef_D_inv = allocate_2D(settings.local_dim);

    /* Set coefficients */
    init_coeffitients(settings);

    /* Set preconditioning operator coefficients */
    init_preconditioning(settings);

    /* Solve Poisson problem for triangle */
    TaskResults res = solve_poisson(settings);
    if (settings.mpi.rank == 0) {
        printf("Elapsed time: %.4lf seconds\n", res.time);
        printf("Total iterations: %d\n", res.iter_num);
    }

    /* Free allocated memory for coefficients */
    free_2D(settings.coef_a, settings.local_dim);
    free_2D(settings.coef_b, settings.local_dim);
    free_2D(settings.coef_F, settings.local_dim);
    free_2D(settings.coef_D_inv, settings.local_dim);

    /* Obtain solution */
    MPI_Barrier(MPI_COMM_WORLD);
    int flatten_size = (settings.local_dim.xdim - 2) * (settings.local_dim.ydim - 2);
    double *flatten = (double *) malloc(flatten_size * sizeof(*flatten));
    if (settings.mpi.rank == 0) {
        /* Allocate memory for global solution */
        double **solution = allocate_2D(settings.global_dim);

        /* Copy 0 rank solution without send and offsets */
        for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
            for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
                solution[y - 1][x - 1] = res.solution[y][x];
            }
        }

        MPI_Status status;
        for (int rank = 1; rank < settings.mpi.size; rank++) {
            /* Receive local dim data from specified rank */
            int local_dim_data[4];
            MPI_Recv(local_dim_data, 4, MPI_INT, rank, rank, MPI_COMM_WORLD, &status);

            /* Receive data from specified rank */
            int xoffset = local_dim_data[0], yoffset = local_dim_data[1];
            int xdim = local_dim_data[2], ydim = local_dim_data[3];
            MPI_Recv(flatten, (xdim - 2) * (ydim - 2), MPI_DOUBLE, rank, rank, MPI_COMM_WORLD, &status);

            /* Copy solution from specified rank */
            for (int y = 0; y < ydim - 2; y++) {
                for (int x = 0; x < xdim - 2; x++) {
                    solution[yoffset + y][xoffset + x] = flatten[y * (xdim - 2) + x];
                }
            }
        }

        /* Print global solution */
        for (int y = 0; y < settings.global_dim.ydim; y++) {
            for (int x = 0; x < settings.global_dim.xdim; x++) {
                printf("%lf ", solution[y][x]);
            }
            putchar('\n');
        }

        /* Free global solution memory */
        free_2D(solution, settings.global_dim);
    } else {
        /* Send local dim data to rank 0 */
        int local_dim_data[4] = {
            settings.offsets.xdim, settings.offsets.ydim,
            settings.local_dim.xdim, settings.local_dim.ydim
        };
        MPI_Send(local_dim_data, 4, MPI_INT, 0, settings.mpi.rank, MPI_COMM_WORLD);

        /* Send data to rank 0 */
        for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
            for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
                flatten[(y - 1) * (settings.local_dim.xdim - 2) + (x - 1)] = res.solution[y][x];
            }
        }
        MPI_Send(flatten, flatten_size, MPI_DOUBLE, 0, settings.mpi.rank, MPI_COMM_WORLD);
    }
    free(flatten);
    free_2D(res.solution, settings.local_dim);

    MPI_Finalize();

    /* Success ! */
    return 0;
}

void get_best_local_dim(TaskSettings *settings) {
    double xy_ratio_global = ((double) settings->global_dim.xdim) / ((double) settings->global_dim.ydim);
    double best_ratio = (double) settings->mpi.size;
    int best_xsize = 1, best_ysize = settings->mpi.size;

    for (int xsize = 1; xsize <= settings->mpi.size; xsize++) {
        int ysize = settings->mpi.size / xsize;
        if (settings->mpi.size % xsize != 0 || xsize < 0.5 * ysize || xsize > 2. * ysize) {
            continue;
        }
        double xdim_local = ((double) settings->global_dim.xdim) / ((double) xsize);
        double ydim_local = ((double) settings->global_dim.ydim) / ((double) ysize);
        double xy_ratio_local = xdim_local / ydim_local;
        if (fabs(xy_ratio_local - xy_ratio_global) < fabs(best_ratio - xy_ratio_global)) {
            best_xsize = xsize;
            best_ysize = ysize;
            best_ratio = xy_ratio_local;
        }
    }
    settings->mpi.ysize = best_ysize;
    settings->mpi.xsize = best_xsize;
    settings->mpi.yrank = settings->mpi.rank / best_xsize;
    settings->mpi.xrank = settings->mpi.rank % best_xsize;

    settings->local_dim.ydim = settings->global_dim.ydim / best_ysize;
    settings->local_dim.xdim = settings->global_dim.xdim / best_xsize;
    int yremain = settings->global_dim.ydim - best_ysize * settings->local_dim.ydim;
    int xremain = settings->global_dim.xdim - best_xsize * settings->local_dim.xdim;
    settings->offsets.ydim = settings->mpi.yrank * settings->local_dim.ydim + DOWNRELU(settings->mpi.yrank, yremain);
    settings->offsets.xdim = settings->mpi.xrank * settings->local_dim.xdim + DOWNRELU(settings->mpi.xrank, xremain);
    settings->local_dim.ydim += 2 + ((settings->mpi.yrank < yremain) ? 1 : 0) ;
    settings->local_dim.xdim += 2 + ((settings->mpi.xrank < xremain) ? 1 : 0) ;
} 

double **allocate_2D(DimSettings dim) {
    double **arr;
    arr = (double **) malloc(dim.ydim * sizeof(*arr));
    for (int y = 0; y < dim.ydim; y++) {
        arr[y] = (double *) calloc(dim.xdim, sizeof(**arr));
    }
    return arr;
}

void free_2D(double **arr, DimSettings dim) {
    for (int row = 0; row < dim.ydim; row++) {
        free(arr[row]);
    }
    free(arr);
}

void init_coeffitients(TaskSettings settings) {
    double xstep = (XRIGHT - XLEFT) / settings.global_dim.xdim;
    double ystep = (YRIGHT - YLEFT) / settings.global_dim.ydim;
    double epsilon = fmax(xstep, ystep) * fmax(xstep, ystep);

    double xoffset = XLEFT + settings.offsets.xdim * xstep;
    double yoffset = YLEFT + settings.offsets.ydim * ystep;

    for (int y = 1; y < settings.local_dim.ydim; y++) {
        for (int x = 1; x < settings.local_dim.xdim; x++) {
            double xleft = xoffset + (x - 0.5) * xstep;
            double xright = xoffset + (x + 0.5) * xstep;

            double yleft = yoffset + (y - 0.5) * ystep;
            double yright = yoffset + (y + 0.5) * ystep;

            /* Get intersection of rectangle and region: coef_F */
            if (x < settings.local_dim.xdim - 1 && y < settings.local_dim.ydim - 1) {
                double x1, x2, y1, y2;
                double x0_bottom, x0_top, integral_bottom, integral_top;
                settings.coef_F[y][x] = 0.;

                /* Left triangle */
                x1 = TO_SEGMENT(xleft, -3., 0.);
                x2 = TO_SEGMENT(xright, x1, 0.);
                y1 = TO_SEGMENT(yleft, 0., 4.);
                y2 = TO_SEGMENT(yright, y1, 4.);

                x0_bottom = TO_SEGMENT(0.75 * y1 - 3., x1, x2);
                x0_top = TO_SEGMENT(0.75 * y2 - 3., x1, x2);

                integral_bottom = INT_LINEAR(4. / 3., 4. - y1, x0_bottom, x2);
                integral_top = INT_LINEAR(4. / 3., 4. - y2, x0_top, x2);
                settings.coef_F[y][x] += (integral_bottom - integral_top);

                /* Right triangle */
                x1 = TO_SEGMENT(xleft, 0., 3.);
                x2 = TO_SEGMENT(xright, x1, 3.);
                y1 = TO_SEGMENT(yleft, 0., 4.);
                y2 = TO_SEGMENT(yright, y1, 4.);

                x0_bottom = TO_SEGMENT(3 - 0.75 * y2, x1, x2);
                x0_top = TO_SEGMENT(3. - 0.75 * y2, x1, x2);
                
                integral_bottom = INT_LINEAR(-4. / 3., 4. - y1, x1, x0_bottom);
                integral_top = INT_LINEAR(-4. / 3., 4. - y2, x1, x0_bottom);
                settings.coef_F[y][x] += (integral_bottom - integral_top);

                /* Normalize intersection area */
                settings.coef_F[y][x] /= (xstep * ystep);
            }

            /* Get intersection of vertical line and region: coef_a */
            if (xleft < -3. || xleft > 3. || yright < 0. || yleft > 4.) {
                settings.coef_a[y][x] = 1. / epsilon;
            } else {
                double y0_top = (xleft < 0.) ? (4. * (1. + xleft / 3.)) : (4. * (1. - xleft / 3.));

                double y1 = TO_SEGMENT(yleft, 0., y0_top);
                double y2 = TO_SEGMENT(yright, 0., y0_top);

                double normalized_len = (y2 - y1) / ystep;
                settings.coef_a[y][x] = normalized_len + (1. - normalized_len) / epsilon;
            }

            /* Get intersection of horizontal line and region: coef_b */
            if (yleft < 0. || yleft > 4. || xright < -3. || xleft > 3.) {
                settings.coef_b[y][x] = 1. / epsilon;
            } else {
                double x0_left = 0.75 * yleft - 3.;
                double x0_right = 3. - 0.75 * yleft;

                double x1 = TO_SEGMENT(xleft, x0_left, x0_right);
                double x2 = TO_SEGMENT(xright, x0_left, x0_right);

                double normalized_len = (x2 - x1) / xstep;
                settings.coef_b[y][x] = normalized_len + (1. - normalized_len) / epsilon;
            }
        }
    }
}

void init_preconditioning(TaskSettings settings) {
    double xstep = (XRIGHT - XLEFT) / settings.global_dim.xdim;
    double ystep = (YRIGHT - YLEFT) / settings.global_dim.ydim;

    for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
        for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
            settings.coef_D_inv[y][x] = 1. / (
                (
                    settings.coef_a[y][x + 1]
                    + settings.coef_a[y][x]
                ) / SQUARE(xstep)
                + (
                    settings.coef_b[y + 1][x]
                    + settings.coef_b[y][x]
                ) / SQUARE(ystep)
            );
        }
    }
}

void copy_2D(double **dst, double **src, DimSettings dim) {
    for (int y = 1; y < dim.ydim - 1; y++) {
        for (int x = 1; x < dim.xdim - 1; x++) {
            dst[y][x] = src[y][x];
        }
    }
}

void apply_preconditioning(double **res, double **vec, TaskSettings settings) {
    for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
        for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
            res[y][x] = settings.coef_D_inv[y][x] * vec[y][x];
        }
    }
}

double get_dot_product(double **u, double **v, double *gammas, TaskSettings settings) {
    double xstep = (XRIGHT - XLEFT) / settings.global_dim.xdim;
    double ystep = (YRIGHT - YLEFT) / settings.global_dim.ydim;

    double dot_local = 0.;
    for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
        for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
            dot_local += u[y][x] * v[y][x];
        }
    }

    MPI_Allgather(&dot_local, 1, MPI_DOUBLE, gammas, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    double res = 0.;
    for (int i = 0; i < settings.mpi.size; i++) {
        res += gammas[i];
    }
    return res * xstep * ystep;
}

void swap_values(double **vec, double *sendbuf, double *recvbuf, TaskSettings settings) {
    int sendrank, recvrank;
    MPI_Request request;
    MPI_Status status;

    if (settings.mpi.ysize > 1) {
        /* Bottom->Up */
        sendrank = (settings.mpi.yrank > 0 ? settings.mpi.yrank - 1 : settings.mpi.ysize - 1) * settings.mpi.xsize + settings.mpi.xrank;
        recvrank = (settings.mpi.yrank < settings.mpi.ysize - 1 ? settings.mpi.yrank + 1 : 0) * settings.mpi.xsize + settings.mpi.xrank;
        if (settings.mpi.yrank > 0) {
            for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
                sendbuf[x] = vec[1][x];
            }
            MPI_Isend(sendbuf, settings.local_dim.xdim, MPI_DOUBLE, sendrank, 0, MPI_COMM_WORLD, &request);
        }
        if (settings.mpi.yrank < settings.mpi.ysize - 1) {
            MPI_Recv(recvbuf, settings.local_dim.xdim, MPI_DOUBLE, recvrank, 0, MPI_COMM_WORLD, &status);
            for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
                vec[settings.local_dim.ydim - 1][x] = recvbuf[x];
            }
        }
        if (settings.mpi.yrank > 0) {
            MPI_Wait(&request, &status);
        }
        
        /* Up->Bottom */
        sendrank = (settings.mpi.yrank < settings.mpi.ysize - 1 ? settings.mpi.yrank + 1 : 0) * settings.mpi.xsize + settings.mpi.xrank;
        recvrank = (settings.mpi.yrank > 0 ? settings.mpi.yrank - 1 : settings.mpi.ysize - 1) * settings.mpi.xsize + settings.mpi.xrank;
        if (settings.mpi.yrank < settings.mpi.ysize - 1) {
            for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
                sendbuf[x] = vec[settings.local_dim.ydim - 2][x];
            }
            MPI_Isend(sendbuf, settings.local_dim.xdim, MPI_DOUBLE, sendrank, 0, MPI_COMM_WORLD, &request);
        }
        if (settings.mpi.yrank > 0) {
            MPI_Recv(recvbuf, settings.local_dim.xdim, MPI_DOUBLE, recvrank, 0, MPI_COMM_WORLD, &status);
            for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
                vec[0][x] = recvbuf[x];
            }
        }
        if (settings.mpi.yrank < settings.mpi.ysize - 1) {
            MPI_Wait(&request, &status);
        }
    }

    if (settings.mpi.xsize > 1) {
        /* Right->Left */
        sendrank = settings.mpi.yrank * settings.mpi.xsize + (settings.mpi.xrank > 0 ? settings.mpi.xrank - 1 : settings.mpi.xsize - 1);
        recvrank = settings.mpi.yrank * settings.mpi.xsize + (settings.mpi.xrank < settings.mpi.xsize - 1 ? settings.mpi.xrank + 1 : 0);
        if (settings.mpi.xrank > 0) {
            for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
                sendbuf[y] = vec[y][1];
            }
            MPI_Isend(sendbuf, settings.local_dim.ydim, MPI_DOUBLE, sendrank, 0, MPI_COMM_WORLD, &request);
        }
        if (settings.mpi.xrank < settings.mpi.xsize - 1) {
            MPI_Recv(recvbuf, settings.local_dim.ydim, MPI_DOUBLE, recvrank, 0, MPI_COMM_WORLD, &status);
            for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
                vec[y][settings.local_dim.xdim - 1] = recvbuf[y];
            }
        }
        if (settings.mpi.xrank > 0) {
            MPI_Wait(&request, &status);
        }
        
        /* Left->Right */
        sendrank = settings.mpi.yrank * settings.mpi.xsize + (settings.mpi.xrank < settings.mpi.xsize - 1 ? settings.mpi.xrank + 1 : 0);
        recvrank = settings.mpi.yrank * settings.mpi.xsize + (settings.mpi.xrank > 0 ? settings.mpi.xrank - 1 : settings.mpi.xsize - 1);
        if (settings.mpi.xrank < settings.mpi.xsize - 1) {
            for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
                sendbuf[y] = vec[y][settings.local_dim.xdim - 2];
            }
            MPI_Isend(sendbuf, settings.local_dim.ydim, MPI_DOUBLE, sendrank, 0, MPI_COMM_WORLD, &request);
        }
        if (settings.mpi.xrank > 0) {
            MPI_Recv(recvbuf, settings.local_dim.ydim, MPI_DOUBLE, recvrank, 0, MPI_COMM_WORLD, &status);
            for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
                vec[y][0] = recvbuf[y];
            }
        }
        if (settings.mpi.xrank < settings.mpi.xsize - 1) {
            MPI_Wait(&request, &status);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

double get_l2_norm(double **vec, double *gammas, TaskSettings settings) {
    return sqrt(get_dot_product(vec, vec, gammas, settings));
}

double get_max_norm(double **vec, double *gammas, TaskSettings settings) {
    double max_local = 0., value = 0.;
    for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
        for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
            if (max_local < (value = fabs(vec[y][x]))) {
                max_local = value;
            }
        }
    }

    MPI_Allgather(&max_local, 1, MPI_DOUBLE, gammas, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    double max_global = 0.;
    for (int i = 0; i < settings.mpi.size; i++) {
        if (gammas[i] > max_global) {
            max_global = gammas[i];
        }
    }
    return max_global;
}

double get_l1_norm(double **vec, double *gammas, TaskSettings settings) {
    double xstep = (XRIGHT - XLEFT) / settings.global_dim.xdim;
    double ystep = (YRIGHT - YLEFT) / settings.global_dim.ydim;

    double l1_local = 0.;
    for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
        for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
            l1_local += fabs(vec[y][x]);
        }
    }

    MPI_Allgather(&l1_local, 1, MPI_DOUBLE, gammas, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    double l1_global = 0.;
    for (int i = 0; i < settings.mpi.size; i++) {
        l1_global += gammas[i];
    }
    return l1_global * xstep * ystep;
}

void apply_laplace_operator(double **res, double **vec, TaskSettings settings) {
    double xstep = (XRIGHT - XLEFT) / settings.global_dim.xdim;
    double ystep = (YRIGHT - YLEFT) / settings.global_dim.ydim;

    double xforward, xbackward, yforward, ybackward;

    for (int y = 1; y < settings.local_dim.ydim - 1; y++) {
        for (int x = 1; x < settings.local_dim.xdim - 1; x++) {
            xforward = settings.coef_a[y][x + 1] * (vec[y][x + 1] - vec[y][x]) / xstep;
            xbackward = settings.coef_a[y][x] * (vec[y][x] - vec[y][x - 1]) / xstep;

            yforward = settings.coef_b[y + 1][x] * (vec[y + 1][x] - vec[y][x]) / ystep;
            ybackward = settings.coef_b[y][x] * (vec[y][x] - vec[y - 1][x]) / ystep;

            res[y][x] = (xbackward - xforward) / xstep + (ybackward - yforward) / ystep;
        }
    }
}

void add_step(double **res, double **vec1, double **vec2, double coef1, double coef2, DimSettings dim) {
    for (int y = 1; y < dim.ydim - 1; y++) {
        for (int x = 1; x < dim.xdim - 1; x++) {
            res[y][x] = coef1 * vec1[y][x] + coef2 * vec2[y][x];
        }
    }
}

TaskResults solve_poisson(TaskSettings settings) {
    double **r_vec, **z_vec, **p_vec, **w_vec, **q_vec;
    double alpha, beta, gamma;
    double l2_norm, max_norm, l1_norm;
    double *gammas, *sendbuf, *recvbuf;

    /* Allocate memory for each vector */
    r_vec = allocate_2D(settings.local_dim);
    z_vec = allocate_2D(settings.local_dim);
    p_vec = allocate_2D(settings.local_dim);
    w_vec = allocate_2D(settings.local_dim);
    q_vec = allocate_2D(settings.local_dim);
    gammas = (double *) malloc(settings.mpi.size * sizeof(*gammas));
    int buf_size = MAX(settings.local_dim.xdim, settings.local_dim.ydim) + settings.mpi.size;
    sendbuf = (double *) malloc(buf_size * sizeof(*sendbuf));
    recvbuf = (double *) malloc(buf_size * sizeof(*recvbuf));

    /* Start clock */
    double t_begin = MPI_Wtime();

    /* Iteration params */
    int iter_num = 0;
    int max_iters = (settings.global_dim.xdim - 2) * (settings.global_dim.ydim - 2);

    /* Perform initial gradient descent step */
    copy_2D(r_vec, settings.coef_F, settings.local_dim); // r_0
    apply_preconditioning(z_vec, r_vec, settings); // z_0
    gamma = get_dot_product(z_vec, r_vec, gammas, settings); // gamma_0
    copy_2D(p_vec, z_vec, settings.local_dim); // p_1

    MPI_Barrier(MPI_COMM_WORLD); // Wait for all vectors
    swap_values(p_vec, sendbuf, recvbuf, settings); // Get border p_vec values from neighbours
    apply_laplace_operator(q_vec, p_vec, settings); // q_1
    alpha = gamma / get_dot_product(q_vec, p_vec, gammas, settings); // alpha_1
    add_step(w_vec, w_vec, p_vec, 1., alpha, settings.local_dim); // w_1

    l2_norm = alpha * get_l2_norm(p_vec, gammas, settings);
    max_norm = alpha * get_max_norm(p_vec, gammas, settings);
    l1_norm = alpha * get_l1_norm(p_vec, gammas, settings);
    if (settings.mpi.rank == 0) {
        printf("Step 0: |w1 - w0| L2 norm=%.6lf, max norm=%.6lf, L1 norm=%.6lf\n", l2_norm, max_norm, l1_norm);
    }

    /* Make some steps until convergence */
    for (iter_num = 1; iter_num <= max_iters && l2_norm >= TOL; iter_num++) {
        add_step(r_vec, r_vec, q_vec, 1., -alpha, settings.local_dim); // r_k
        apply_preconditioning(z_vec, r_vec, settings); // z_k
        beta = 1. / gamma; // 1 / gamma_{k-1}
        gamma = get_dot_product(z_vec, r_vec, gammas, settings); // gamma_k
        add_step(p_vec, z_vec, p_vec, 1., beta * gamma, settings.local_dim); // p_{k+1}

        MPI_Barrier(MPI_COMM_WORLD); // Wait for all vectors
        swap_values(p_vec, sendbuf, recvbuf, settings); // Get border p_vec values from neighbours
        apply_laplace_operator(q_vec, p_vec, settings); // q_{k+1}
        alpha = gamma / get_dot_product(q_vec, p_vec, gammas, settings); // alpha_{k+1}
        add_step(w_vec, w_vec, p_vec, 1., alpha, settings.local_dim);

        l2_norm = alpha * get_l2_norm(p_vec, gammas, settings);
        max_norm = get_max_norm(p_vec, gammas, settings);
        l1_norm = alpha * get_l1_norm(p_vec, gammas, settings);
        if (settings.mpi.rank == 0) {
            printf(
                "Step %d: |w%d - w%d| L2 norm=%.6lf, max norm=%.6lf, L1 norm=%.6lf\n",
                iter_num,
                iter_num + 1,
                iter_num,
                l2_norm,
                max_norm,
                l1_norm
            );
        }
    }

    /* End clock */
    double t_end = MPI_Wtime();

    /* Free all allocated memory, except solution */
    free_2D(r_vec, settings.local_dim);
    free_2D(z_vec, settings.local_dim);
    free_2D(p_vec, settings.local_dim);
    free_2D(q_vec, settings.local_dim);
    free(gammas);
    free(recvbuf);
    free(sendbuf);

    /* Return TaskResults */
    TaskResults res;
    res.solution = w_vec;
    res.iter_num = iter_num;
    res.time = t_end - t_begin;
    return res;
}
