#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct {
    int degree;
    double *coefficients;
} Polynomial;

Polynomial *generate_random_polynomial(int max_degree, double range);
void print_polynomial(Polynomial *poly);
void partial_fraction_decomposition(Polynomial *numerator, Polynomial *denominator);
void free_polynomial(Polynomial *poly);

int main() {
    srand(time(NULL));

    Polynomial *numerator, *denominator;

    numerator = generate_random_polynomial(20000, 20000);
    denominator = generate_random_polynomial(20000, 20000);

    printf("Generated numerator polynomial:\n");
    print_polynomial(numerator);

    printf("Generated denominator polynomial:\n");
    print_polynomial(denominator);

    clock_t start = clock();
    printf("The partial fraction decomposition is:\n");
    partial_fraction_decomposition(numerator, denominator);
    clock_t end = clock();
    double execution_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nExecution time: %lf seconds\n", execution_time);

    free_polynomial(numerator);
    free_polynomial(denominator);

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

void print_polynomial(Polynomial *poly) {
    for (int i = 0; i <= poly->degree; i++) {
        printf("%+.2lf", poly->coefficients[i]);
        if (i < poly->degree) {
            printf("x^%d ", poly->degree - i);
        }
    }
    printf("\n");
}

void partial_fraction_decomposition(Polynomial *numerator, Polynomial *denominator) {
    if (denominator->degree == 1) {
        double A = numerator->coefficients[numerator->degree] / denominator->coefficients[0];
        printf("%+.2lfx", A);
    } else {
        for (int i = 0; i < denominator->degree; i++) {
            double x = -denominator->coefficients[i + 1] / denominator->coefficients[0];
            double A = numerator->coefficients[numerator->degree];

            for (int j = 1; j <= numerator->degree; j++) {
                A += numerator->coefficients[numerator->degree - j] * pow(x, j);
            }

            printf("(%+.2lf)/(x%+.2lf)", A, -x);
            if (i < denominator->degree - 1) {
                printf(" + ");
            }
        }
    }
}

void free_polynomial(Polynomial *poly) {
    free(poly->coefficients);
    free(poly);
}
