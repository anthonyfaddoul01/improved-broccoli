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

Polynomial *generate_random_polynomial(int max_degree, double range);
void broadcast_polynomial(Polynomial *poly, int rank);
void print_polynomial(Polynomial *poly);
void parallel_partial_fraction_decomposition(Polynomial *numerator, Polynomial *denominator, int rank, int size);
void free_polynomial(Polynomial *poly);

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);

    Polynomial *numerator, *denominator;

    if (rank == 0) {
        numerator = generate_random_polynomial(10000, 10000);
        denominator = generate_random_polynomial(10000, 10000);
    } else {
        numerator = (Polynomial *)malloc(sizeof(Polynomial));
        denominator = (Polynomial *)malloc(sizeof(Polynomial));
    }

    broadcast_polynomial(numerator, rank);
    broadcast_polynomial(denominator, rank);

    if (rank == 0) {
        printf("Generated numerator polynomial:\n");
        print_polynomial(numerator);
        printf("Generated denominator polynomial:\n");
        print_polynomial(denominator);
    }

    parallel_partial_fraction_decomposition(numerator, denominator, rank, size);

    free_polynomial(numerator);
    free_polynomial(denominator);

    MPI_Finalize();

    return 0;
}

Polynomial *generate_random_polynomial(int max_degree, double range) {
    int degree = rand() % (max_degree - 1) + 1;

    Polynomial *poly = (Polynomial *)malloc(sizeof(Polynomial));
    poly->degree = degree;
    poly->coefficients = (double *)malloc((degree + 1) * sizeof(double));

    for (int i = 0; i <= degree; i++) {
        poly->coefficients[i] = ((double)rand() / RAND_MAX) * 2 * range - range;
    }

    return poly;
}

void broadcast_polynomial(Polynomial *poly, int rank) {
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (rank == 0) {
        for (int i = 1; i < comm_size; i++) {
            MPI_Send(&poly->degree, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&poly->degree, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        poly->coefficients = (double *)malloc((poly->degree + 1) * sizeof(double));
    }

    MPI_Bcast(poly->coefficients, poly->degree + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

void parallel_partial_fraction_decomposition(Polynomial *numerator, Polynomial *denominator, int rank, int size) {
    int local_start, local_end;
    int range = denominator->degree / size;
    local_start = rank * range;
    
    local_end = (rank == size - 1) ? denominator->degree : (rank + 1) * range - 1;
    char *local_result = malloc(1024);
    local_result[0] = '\0';

    for (int i = local_start; i <= local_end; i++) {
        double x = -denominator->coefficients[i + 1] / denominator->coefficients[0];
        double A = numerator->coefficients[numerator->degree];

        for (int j = 1; j <= numerator->degree; j++) {
            A += numerator->coefficients[numerator->degree - j] * pow(x, j);
        }

        char term[128];
        sprintf(term, "(%+.2lf)/(x%+.2lf)", A, -x);
        strcat(local_result, term);
        if (i < denominator->degree - 1) {
            strcat(local_result, " + ");
        }
    }

    // Gather the partial results
    char *results = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        results = malloc(size * 1024 * sizeof(char));
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }

    int local_result_len = strlen(local_result);
    MPI_Gather(&local_result_len, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    MPI_Gatherv(local_result, local_result_len, MPI_CHAR, results, recvcounts, displs, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The partial fraction decomposition is:\n%s\n", results);
        free(results);
        free(recvcounts);
        free(displs);
    }

    free(local_result);
}

void free_polynomial(Polynomial *poly) {
    free(poly->coefficients);
    free(poly);
}