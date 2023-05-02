#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <string.h>


typedef struct {
    int degree;
    double *coefficients;
} Polynomial;

Polynomial *generate_random_polynomial(int max_degree, double range, int rank, int size) {
    int degree = rand() % (max_degree - 1) + 1;

    Polynomial *poly = (Polynomial *)malloc(sizeof(Polynomial));
    poly->degree = degree;
    poly->coefficients = (double *)malloc((degree + 1) * sizeof(double));

    for (int i = rank; i <= degree; i += size) {
        poly->coefficients[i] = ((double)rand() / RAND_MAX) * 2 * range - range;
    }

    return poly;
}

void print_polynomial(Polynomial *poly) {
    for (int i = 0; i <= poly->degree; i++) {
        printf("%+.2lf", poly->coefficients[i]);
        if (i < poly->degree) {
            printf("x^%d ", poly->degree - i);
        }
    }
    printf("\n");
}

char *partial_fraction_decomposition(Polynomial *numerator, Polynomial *denominator) {
    char *result = malloc(1024);
    result[0] = '\0';

    if (denominator->degree == 1) {
        double A = numerator->coefficients[numerator->degree] / denominator->coefficients[0];
        sprintf(result, "%+.2lfx", A);
    } else {
        for (int i = 0; i < denominator->degree; i++) {
            double x = -denominator->coefficients[i + 1] / denominator->coefficients[0];
            double A = numerator->coefficients[numerator->degree];

            for (int j = 1; j <= numerator->degree; j++) {
                A += numerator->coefficients[numerator->degree - j] * pow(x, j);
            }

            char term[128];
            sprintf(term, "(%+.2lf)/(x%+.2lf)", A, -x);
            strcat(result, term);
            if (i < denominator->degree - 1) {
                strcat(result, " + ");
            }
        }
    }

    return result;
}


void free_polynomial(Polynomial *poly) {
    free(poly->coefficients);
    free(poly);
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);

    Polynomial *numerator, *denominator;

    numerator = generate_random_polynomial(10000, 10000, rank, size);
    denominator = generate_random_polynomial(10000, 10000, rank, size);

    double *full_coefficients = (double *)malloc((numerator->degree + 1) * sizeof(double));

    for (int i = 0; i <= numerator->degree; i++) {
        MPI_Gather(&(numerator->coefficients[i]), 1, MPI_DOUBLE, full_coefficients, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            numerator->coefficients[i] = full_coefficients[i];
        }
    }

    

    for (int i = 0; i <= denominator->degree; i++) {
        MPI_Gather(&(denominator->coefficients[i]), 1, MPI_DOUBLE, full_coefficients, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                if (rank == 0) {
            denominator->coefficients[i] = full_coefficients[i];
        }
    }

    if (rank == 0) {
    printf("Generated numerator polynomial:\n");
    print_polynomial(numerator);
    printf("Generated denominator polynomial:\n");
    print_polynomial(denominator);

    char *decomposition = partial_fraction_decomposition(numerator, denominator);
    printf("The partial fraction decomposition is:\n%s\n", decomposition);
    free(decomposition);
}



    free(full_coefficients);
    free_polynomial(numerator);
    free_polynomial(denominator);

    MPI_Finalize();

    return 0;
}


