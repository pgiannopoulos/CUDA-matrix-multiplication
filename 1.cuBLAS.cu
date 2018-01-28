/*Αλγόριθμος υπολογισμού γινομένου μητρώων, A^t * A = C, όπου Α (N*M), A^t (M*N) και C (M*M),
με χρήση της βιβλιοθήκης cuBLAS.*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include "cublas.h"
#include "cublas_v2.h"

/*Μακρο-εντολή για έλεγχο λαθών*/
#define cudaCheckError() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}
/*Εκτύπωση των στοιχείων του μητρώου Α και των C_h, C που αποτελούν τα αποτελέσματα των γινομένων στην CPU και στην GPU αντίστοιχα.*/
void printMatrices(double *A_h, double *C_h, double *C, int M, int N)
{
	printf("Matrix A:\n");
	for (int i = 0; i < N*M; i++) {
		printf("%lf   ", A_h[i]);
		if ((i + 1) % M == 0)
			printf("\n");
	}
	printf("Multiplication Result on GPU:\n");
	for (int i = 0; i < M*M; i++) {
		printf("%lf   ", C_h[i]);
		if ((i + 1) % M == 0)
			printf("\n");
	}
	printf("Multiplication Result on CPU:\n");
	for (int i = 0; i < M*M; i++) {
		printf("%lf   ", C[i]);
		if ((i + 1) % M == 0)
			printf("\n");
	}
}
/*Υπολογισμός του ανάστροφου μητρώου Α.*/
double *transpose(double *A, int Α_rows, int Α_columns) {
	double *At_hranspose;
	int size = Α_rows * Α_columns * sizeof(double);
	At_hranspose = (double *)malloc(size);
	if ((At_hranspose == NULL)) {
		printf("Could not allocate memory.\n");
		exit(0);
	}
	int k = 0;
	for (int i = 0; i < Α_columns; i++)
		for (int j = 0; j < Α_rows; j++) {
			At_hranspose[k] = A[j*Α_columns + i];
			k++;
		}
	return At_hranspose;
}
/*Υπολογισμός του πολλαπλασιασμού 2 μητρώων στη CPU(host), A * B = C.*/
double *MatrixMulOnHost(double *A, double *B, int B_rows, int B_columns)
{
	double *C;
	int size = B_columns * B_columns * sizeof(double);
	C = (double *)malloc(size);
	if ((C == NULL)) {
		printf("Could not allocate memory.\n");
		exit(0);
	}
	for (int i = 0; i < B_columns; i++) {
		for (int j = 0; j < B_columns; j++) {
			double sum = 0;
			for (int k = 0; k < B_rows; k++)
			{
				double a = A[i*B_rows + k];
				double b = B[k*B_columns + j];
				sum += a*b;
			}
			C[i*B_columns + j] = sum;
		}
	}
	return C;
}
static __inline__ void cuBlasmul(int m, int n, int k, double *A, double *B, double *C)
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasOperation_t transa = CUBLAS_OP_N;
	cublasOperation_t transb = CUBLAS_OP_T;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	int lda = m, ldb = n, ldc = m;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaCheckError();
	cudaEventCreate(&stop);
	cudaCheckError();
	float milliseconds = 0;

	cudaEventRecord(start);
	cudaCheckError();
	cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	cudaEventRecord(stop);
	cudaCheckError();
	cudaEventSynchronize(stop);
	cudaCheckError();
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaCheckError();
	printf("Time Elapsed:%f \n", milliseconds);


}
/*Υπολογισμός του γινομένου των μητρώων στη GPU και στη CPU για διασταύρωση των αποτελεσμάτων*/
int main(int argc, char* argv[])
{

	double *A_h, *C_h;	//Δείκτες για μνήμη στον host
	double *A_d, *C_d; 	//Δείκτες για μνήμη στο device
	double *At_h;	//Δείκτης για μνήμη στον host για τον ανάστροφο του Α
	double *C;		//Δείκτης για μνήμη στον host για τον υπολογισμό του γινομένου στη CPU
	int N, M;	//Διαστάσεις του μητρώου Α (Ν*Μ)
	int size_A, size_C;	//Μέγεθη των μητρώων στη μνήμη
	int i, j;	//Μετρητές
				

	if (argc != 3) {
		printf("Provide the problem size.\n");
		//exit(0);
	}

	N = 1024;
	M = 1024;

	size_A = N * M * sizeof(double);
	size_C = M * M * sizeof(double);

	A_h = (double *)malloc(size_A);
	At_h = (double *)malloc(size_A);
	C_h = (double *)malloc(size_C);
	C_d = (double *)malloc(size_C);
	C = (double *)malloc(size_C);

	if ((A_h == NULL) || (At_h == NULL) || (C_h == NULL) || (C_d == NULL) || (C == NULL)) {
		printf("Could not allocate memory.\n");
		exit(0);
	}
	//Αρχικοποίηση του μητρώου Α με τυχαίες τιμές.
	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			A_h[i*M + j] = rand() / (double)RAND_MAX;
		}
	}

	// Δέσμευση μνήμης στο device για τα διανύσματα
	cudaMalloc((void **)&A_d, size_A);
	cudaCheckError();
	//cudaMalloc((void **)&B_d, size);
	cudaMalloc((void **)&C_d, size_C);
	cudaCheckError();

	//Αντιγραφή A_h στο device
	cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
	cudaCheckError();

	cuBlasmul(M, M, N, A_d, A_d, C_d);
	cudaCheckError();

	//Αντιγραφή του αποτελέσματος C_d στον host
	cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);
	cudaCheckError();

	//Αποδέσμευση μνήμης στο device
	cudaFree(A_d);
	cudaCheckError();
	cudaFree(C_d);
	cudaCheckError();

	//Υπολογισμός του γινομένου στη CPU, Α_t * A_h = C.
	At_h = transpose(A_h, N, M);
	C = MatrixMulOnHost(At_h, A_h, N, M);

	//Εκτύπωση των μητρώων A_h, C_h και C.
	//printMatrices(A_h, C_h, C, M, N);

	//Αξιολόγηση της ορθότητας του πολλαπλασιασμού στη GPU.
	for (i = 0; i < M*M; i++) {
		if (C[i] - C_h[i] > 0.000001) {
			printf("The matrix multiplication on GPU was unsuccessful!\n");
			exit(0);
		}
	}
	printf("The matrix multiplication on GPU was successful!\n");
}