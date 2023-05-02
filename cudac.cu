#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

typedef struct {
    int degree;
    double *coefficients;
} Polynomial;


__global__ void initialize_curand(curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(clock64(), idx, 0, &state[idx]);
}

__global__ void generate_random_polynomial_kernel(int max_degree, double range, Polynomial *poly, curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx <= poly->degree) {
        double rand_val = curand_uniform_double(&state[idx]);
        poly->coefficients[idx] = (rand_val * 2 * range - range);
    }
}

Polynomial *generate_random_polynomial(int max_degree, double range) {
    int degree = rand() % (max_degree - 1) + 1;

    Polynomial *poly = (Polynomial *)malloc(sizeof(Polynomial));
    poly->degree = degree;

    cudaMalloc((void **)&poly->coefficients, (degree + 1) * sizeof(double));

    curandState *state;
    cudaMalloc((void **)&state, (degree + 1) * sizeof(curandState));

    int blockSize = 256;
    int gridSize = (degree + blockSize - 1) / blockSize;

    initialize_curand<<<gridSize, blockSize>>>(state);
    generate_random_polynomial_kernel<<<gridSize, blockSize>>>(max_degree, range, poly, state);

    cudaFree(state);

    return poly;
}



__global__ void print_polynomial_kernel(Polynomial *poly, double *coefficients_host) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx <= poly->degree) {
        coefficients_host[idx] = poly->coefficients[idx];
    }
}

void print_polynomial(Polynomial *poly) {
    double *coefficients_host = (double *)malloc((poly->degree + 1) * sizeof(double));
    
    int blockSize = 256;
    int gridSize = (poly->degree + blockSize - 1) / blockSize;
    
    print_polynomial_kernel<<<gridSize, blockSize>>>(poly, coefficients_host);
    
    cudaDeviceSynchronize();

    for (int i = 0; i <= poly->degree; i++) {
        printf("%+.2lf", coefficients_host[i]);
        if (i < poly->degree) {
            printf("x^%d ", poly->degree - i);
        }
    }
    printf("\n");
    
    free(coefficients_host);
}

void partial_fraction_decomposition(Polynomial *numerator, Polynomial *denominator) {
    // Remain same as original function as this is not ideal for GPU computation
}

void free_polynomial(Polynomial *poly) {
    cudaFree(poly->coefficients);
    free(poly);
}

int main() {
    srand(time(NULL));

    Polynomial *numerator, *denominator;

    numerator = generate_random_polynomial(200000, 200000);
    denominator = generate_random_polynomial(200000, 200000);

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

