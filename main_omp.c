#include <malloc.h>
#include <math.h>
#include <omp.h>
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

enum {
    ERR_SUCCESS = 0,
    ERR_MALLOC = 1,
    ERR_INPUT = 2,
    ERR_BAD_ARGS = 3,
    ERR_IO = 4
};

void init_coeffitients(
    double *coef_a,
    double *coef_b,
    double *coef_F, 
    int xdim, 
    int ydim
);

void init_preconditioning(
    double *coef_a,
    double *coef_b,
    double *coef_D_inv,
    int xdim,
    int ydim
);

int solve_poisson(
    double *coef_a,
    double *coef_b,
    double *coef_F,
    double *coef_D_inv,
    int xdim,
    int ydim,
    double **solution
);

void apply_preconditioning(
    double *coef_D_inv,
    double *vec,
    double *res,
    int dim
);

void apply_laplace_operator(
    double *coef_a,
    double *coef_b,
    double *vec,
    double *res,
    int xdim,
    int ydim
);

double get_dot_product(
    double *vec1,
    double *vec2,
    double xstep,
    double ystep,
    int dim
);

double get_l2_norm(
    double *vec,
    int dim
);

double get_l1_norm(
    double *vec,
    int dim
);

double get_max_norm(
    double *vec,
    int dim
);

void add_step_inplace(
    double *res,
    double *vec,
    double coef,
    int dim
);

void add_step(
    double *res,
    double *vec1,
    double *vec2,
    double coef1,
    double coef2,
    int dim
);

int main(int argc, char **argv) {

    /* Read grid parameters */
    if (argc < 4) {
        fprintf(stderr, "Error reading grid parameters\nUsage: %s xdim ydim num_threads\n", argv[0]);
        return ERR_INPUT;
    }
    int xdim = atoi(argv[1]), ydim = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    if (xdim < 4 || ydim  < 4) {
        fprintf(stderr, "Bad params error. Grid parameters should be greater than 3\n");
        return ERR_BAD_ARGS;
    }
    if (num_threads < 1 || num_threads > 32) {
        fprintf(stderr, "Bad params error. Number of threads should be between 1 and 32");
        return ERR_BAD_ARGS;
    }
    omp_set_num_threads(num_threads);

    /* Allocate memory for coefficients */
    double *coef_a, *coef_b, *coef_F;
    if ((coef_a = (double *) malloc(xdim * ydim * sizeof(*coef_a))) == NULL) {
        fprintf(stderr, "Error allocating memory for coef_a\n");
        return ERR_MALLOC;
    }
    if ((coef_b = (double *) malloc(xdim * ydim * sizeof(*coef_b))) == NULL) {
        fprintf(stderr, "Error allocating memory for coef_b\n");
        free(coef_a);
        return ERR_MALLOC;
    }
    if ((coef_F = (double *) malloc((xdim - 1) * (ydim - 1) * sizeof(*coef_F))) == NULL) {
        fprintf(stderr, "Error allocating memory for coef_F\n");
        free(coef_a), free(coef_b);
        return ERR_MALLOC;
    }

    /* Set coefficients */
    init_coeffitients(coef_a, coef_b, coef_F, xdim, ydim);

    /* Allocate memory for preconditioning operator */
    double *coef_D_inv = (double *) malloc((xdim - 1) * (ydim - 1) * sizeof(*coef_D_inv));
    if (coef_D_inv == NULL) {
        fprintf(stderr, "Error allocating memory for coef_D_inv\n");
        free(coef_a), free(coef_b), free(coef_F);
        return ERR_MALLOC;
    }

    /* Set preconditioning operator coefficients */
    init_preconditioning(coef_a, coef_b, coef_D_inv, xdim, ydim);

    /* Solve Poisson problem for triangle */
    double *solution;
    int err_code = solve_poisson(coef_a, coef_b, coef_F, coef_D_inv, xdim, ydim, &solution);
    
    /* Free all allocated memory */
    free(coef_a), free(coef_b), free(coef_F), free(coef_D_inv);

    /* Check error in solution */
    if (err_code != ERR_SUCCESS) {
        return err_code;
    }

    /* Save result to file */
    for (int h = 0; h <= ydim; h++) {
        for (int w = 0; w <= xdim; w++) {
            if (h == 0 || w == 0 || h == ydim || w == xdim) {
                printf("%lf ", 0.);
            } else {
                printf("%lf ", solution[(h - 1) * (xdim - 1) + (w - 1)]);
            }
        }
        putchar('\n');
    }
    free(solution);
    
    /* Success ! */
    return ERR_SUCCESS;
}

void init_coeffitients(
    double *coef_a,
    double *coef_b,
    double *coef_F, 
    int xdim, 
    int ydim
) {
    double xstep = (XRIGHT - XLEFT) / xdim;
    double ystep = (YRIGHT - YLEFT) / ydim;
    double epsilon = fmax(xstep, ystep) * fmax(xstep, ystep);

    #pragma omp parallel for
    for (int idx = 0; idx < xdim * ydim; idx++) {
        double w = idx % xdim;
        double xleft = XLEFT + (w + 0.5) * xstep;
        double xright = XLEFT + (w + 1.5) * xstep;

        double h = idx / xdim;        
        double yleft = YLEFT + (h + 0.5) * ystep;
        double yright = YLEFT + (h + 1.5) * ystep;

        /* Get intersection of rectangle and region: coef_F */
        if (w < xdim - 1 && h < ydim - 1) {
            int idx_local = h * (xdim - 1) + w;
            double x1, x2, y1, y2;
            double x0_bottom, x0_top, integral_bottom, integral_top;
            coef_F[idx_local] = 0.;

            /* Left triangle */
            x1 = TO_SEGMENT(xleft, -3., 0.);
            x2 = TO_SEGMENT(xright, x1, 0.);
            y1 = TO_SEGMENT(yleft, 0., 4.);
            y2 = TO_SEGMENT(yright, y1, 4.);

            x0_bottom = TO_SEGMENT(0.75 * y1 - 3., x1, x2);
            x0_top = TO_SEGMENT(0.75 * y2 - 3., x1, x2);

            integral_bottom = INT_LINEAR(4. / 3., 4. - y1, x0_bottom, x2);
            integral_top = INT_LINEAR(4. / 3., 4. - y2, x0_top, x2);
            coef_F[idx_local] += (integral_bottom - integral_top);

            /* Right triangle */
            x1 = TO_SEGMENT(xleft, 0., 3.);
            x2 = TO_SEGMENT(xright, x1, 3.);
            y1 = TO_SEGMENT(yleft, 0., 4.);
            y2 = TO_SEGMENT(yright, y1, 4.);

            x0_bottom = TO_SEGMENT(3 - 0.75 * y2, x1, x2);
            x0_top = TO_SEGMENT(3. - 0.75 * y2, x1, x2);
            
            integral_bottom = INT_LINEAR(-4. / 3., 4. - y1, x1, x0_bottom);
            integral_top = INT_LINEAR(-4. / 3., 4. - y2, x1, x0_bottom);
            coef_F[idx_local] += (integral_bottom - integral_top);

            /* Normalize intersection area */
            coef_F[idx_local] /= (xstep * ystep);
        }

        /* Get intersection of vertical line and region: coef_a */
        if (xleft < -3. || xleft > 3. || yright < 0. || yleft > 4.) {
            coef_a[idx] = 1. / epsilon;
        } else {
            double y0_top = (xleft < 0.) ? (4. * (1. + xleft / 3.)) : (4. * (1. - xleft / 3.));

            double y1 = TO_SEGMENT(yleft, 0., y0_top);
            double y2 = TO_SEGMENT(yright, 0., y0_top);

            double normalized_len = (y2 - y1) / ystep;
            coef_a[idx] = normalized_len + (1. - normalized_len) / epsilon;
        }

        /* Get intersection of horizontal line and region: coef_b */
        if (yleft < 0. || yleft > 4. || xright < -3. || xleft > 3.) {
            coef_b[idx] = 1. / epsilon;
        } else {
            double x0_left = 0.75 * yleft - 3.;
            double x0_right = 3. - 0.75 * yleft;

            double x1 = TO_SEGMENT(xleft, x0_left, x0_right);
            double x2 = TO_SEGMENT(xright, x0_left, x0_right);

            double normalized_len = (x2 - x1) / xstep;
            coef_b[idx] = normalized_len + (1. - normalized_len) / epsilon;
        }
    }
}

void init_preconditioning(
    double *coef_a,
    double *coef_b,
    double *coef_D_inv,
    int xdim,
    int ydim
) {
    double xstep = (XRIGHT - XLEFT) / xdim;
    double ystep = (YRIGHT - YLEFT) / ydim;

    #pragma omp parallel for
    for (int idx = 0; idx < (xdim - 1) * (ydim - 1); idx++) {
        int w = idx % (xdim - 1);
        int h = idx / (xdim - 1);

        coef_D_inv[idx] = 1. / (
            (
                coef_a[h * xdim + w + 1]
                + coef_a[h * xdim + w]
            ) / SQUARE(xstep)
            + (
                coef_b[(h + 1) * xdim + w]
                + coef_b[h * xdim + w]
            ) / SQUARE(ystep)
        );
    }
}

int solve_poisson(
    double *coef_a,
    double *coef_b,
    double *coef_F,
    double *coef_D_inv,
    int xdim,
    int ydim,
    double **solution
) {
    double *r_vec, *z_vec, *p_vec, *w_vec, *q_vec;
    double alpha, beta, gamma;
    double l2_norm, max_norm, l1_norm;

    size_t vec_size = (xdim - 1) * (ydim - 1);
    double xstep = (XRIGHT - XLEFT) / xdim;
    double ystep = (YRIGHT - YLEFT) / ydim;

    /* Allocate memory for each vector */
    if ((r_vec = malloc(vec_size * sizeof(*r_vec))) == NULL) {
        fprintf(stderr, "Can't allocate memory for r_vec\n");
        return ERR_MALLOC;
    }

    if ((z_vec = malloc(vec_size * sizeof(*z_vec))) == NULL) {
        fprintf(stderr, "Can't allocate memory for z_vec\n");
        free(r_vec);
        return ERR_MALLOC;
    }

    if ((p_vec = malloc(vec_size * sizeof(*p_vec))) == NULL) {
        fprintf(stderr, "Can't allocate memory for p_vec\n");
        free(r_vec), free(z_vec);
        return ERR_MALLOC;
    }

    if ((w_vec = calloc(vec_size, sizeof(*w_vec))) == NULL) {
        fprintf(stderr, "Can't allocate memory for w_vec\n");
        free(r_vec), free(z_vec), free(p_vec);
        return ERR_MALLOC;
    }

    if ((q_vec = malloc(vec_size * sizeof(*q_vec))) == NULL) {
        fprintf(stderr, "Can't allocate memory for q_vec\n");
        free(r_vec), free(z_vec), free(p_vec), free(w_vec);
        return ERR_MALLOC;
    }

    /* Start clock */
    double t_begin = omp_get_wtime();

    /* Perform initial gradient descent step */
    memcpy(r_vec, coef_F, vec_size * sizeof(*r_vec)); // r_0
    apply_preconditioning(coef_D_inv, r_vec, z_vec, vec_size); // z_0
    gamma = get_dot_product(z_vec, r_vec, xstep, ystep, vec_size); // gamma_0
    memcpy(p_vec, z_vec, vec_size * sizeof(*p_vec)); // p_1
    apply_laplace_operator(coef_a, coef_b, p_vec, q_vec, xdim, ydim); // q_1
    alpha = gamma / get_dot_product(q_vec, p_vec, xstep, ystep, vec_size); // alpha_1
    add_step_inplace(w_vec, p_vec, alpha, vec_size); // w_1

    l2_norm = alpha * sqrt(xstep * ystep) * get_l2_norm(p_vec, vec_size);
    max_norm = alpha * get_max_norm(p_vec, vec_size);
    l1_norm = alpha * xstep * ystep * get_l1_norm(p_vec, vec_size);
    printf("Step 0: |w1 - w0| L2 norm=%.6lf, max norm=%.6lf, L1 norm=%.6lf\n", l2_norm, max_norm, l1_norm);

    /* Make some steps until convergence */
    for (int step_num = 1; step_num <= ((int) vec_size) && l2_norm >= TOL; step_num++) {
        add_step_inplace(r_vec, q_vec, -alpha, vec_size); // r_k
        apply_preconditioning(coef_D_inv, r_vec, z_vec, vec_size); // z_k
        beta = 1 / gamma; // 1 / gamma_{k-1}
        gamma = get_dot_product(z_vec, r_vec, xstep, ystep, vec_size); // gamma_k
        add_step(p_vec, z_vec, p_vec, 1., beta * gamma, vec_size); // p_{k+1}
        apply_laplace_operator(coef_a, coef_b, p_vec, q_vec, xdim, ydim); // q_{k+1}
        alpha = gamma / get_dot_product(q_vec, p_vec, xstep, ystep, vec_size); // alpha_{k+1}
        add_step_inplace(w_vec, p_vec, alpha, vec_size); // w_{k+1}

        l2_norm = alpha * sqrt(xstep * ystep) * get_l2_norm(p_vec, vec_size);
        max_norm = alpha * get_max_norm(p_vec, vec_size);
        l1_norm = alpha * xstep * ystep * get_l1_norm(p_vec, vec_size);
        printf(
            "Step %d: |w%d - w%d| L2 norm=%.6lf, max norm=%.6lf, L1 norm=%.6lf\n",
            step_num,
            step_num + 1,
            step_num,
            l2_norm,
            max_norm,
            l1_norm
        );
    }

    /* End clock */
    double t_end = omp_get_wtime();
    printf("Convergence reached. Elapsed time: %.3lf seconds\n", t_end - t_begin);

    /* Free all allocated memory, except solution */
    free(r_vec), free(z_vec), free(p_vec), free(q_vec);
    *solution = w_vec;
    return ERR_SUCCESS;
}

void apply_preconditioning(
    double *coef_D_inv,
    double *vec,
    double *res,
    int dim
) {
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        res[i] = coef_D_inv[i] * vec[i];
    }
}

void apply_laplace_operator(
    double *coef_a,
    double *coef_b,
    double *vec,
    double *res,
    int xdim,
    int ydim
) {
    double xstep = (XRIGHT - XLEFT) / xdim;
    double ystep = (YRIGHT - YLEFT) / ydim;
    double center, xforward, xbackward, yforward, ybackward;
    int w, h;

    #pragma omp parallel for private(center, xforward, xbackward, yforward, ybackward, w, h)
    for (int idx = 0; idx < (xdim - 1) * (ydim - 1); idx++) {
        w = idx % (xdim - 1);
        h = idx / (xdim - 1);

        center = vec[idx];
        xforward = (w < (xdim - 2) ? vec[idx + 1] : 0.);
        xbackward = (w > 0 ? vec[idx - 1] : 0.);
        yforward = (h < (ydim - 2) ? vec[idx + (xdim - 1)] : 0.);
        ybackward = (h > 0 ? vec[idx - (xdim - 1)] : 0.);

        xforward = coef_a[h * xdim + w + 1] * (xforward - center) / xstep;
        xbackward = coef_a[h * xdim + w] * (center - xbackward) / xstep;
        yforward = coef_b[(h + 1) * xdim + w] * (yforward - center) / ystep;
        ybackward = coef_b[h * xdim + w] * (center - ybackward) / ystep;
        res[idx] = (xbackward - xforward) / xstep + (ybackward - yforward) / ystep;
    }
}

double get_dot_product(
    double *vec1,
    double *vec2,
    double xstep,
    double ystep,
    int dim
) {
    double res = 0.;
    #pragma omp parallel for reduction(+: res)
    for (int i = 0; i < dim; i++) {
        res += vec1[i] * vec2[i];
    }
    return res * xstep * ystep;
}

double get_l1_norm(
    double *vec,
    int dim
) {
    double res = 0.;
    #pragma omp parallel for reduction(+: res)
    for (int i = 0; i < dim; i++) {
        res += fabs(vec[i]);
    }
    return res;
}

double get_l2_norm(
    double *vec,
    int dim
) {
    double res = 0.;
    #pragma omp parallel for reduction(+: res)
    for (int i = 0; i < dim; i++) {
        res += SQUARE(vec[i]);
    }
    return sqrt(res);
}

double get_max_norm(
    double *vec,
    int dim
) {
    double res = 0., value = 0.;
    for (int i = 0; i < dim; i++) {
        if (res < (value = fabs(vec[i]))) {
            res = value;
        }
    }
    return res;
}

void add_step_inplace(
    double *res,
    double *vec,
    double coef,
    int dim
) {
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        res[i] += coef * vec[i];
    }
}

void add_step(
    double *res,
    double *vec1,
    double *vec2,
    double coef1,
    double coef2,
    int dim
) {
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        res[i] = coef1 * vec1[i] + coef2 * vec2[i];
    }
}

