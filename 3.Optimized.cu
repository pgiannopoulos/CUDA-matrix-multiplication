/*����������� ���������� ����������� ��������� �������, A^t * A = C, ���� � (N*M), A^t (M*N) ��� C (M*M).*/
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#define TILE_DIM 16		//Tile dimension
#define BLOCK_SIZE_PER_DIM 16	//Block dimension		


/*�����-������ ��� ������ �����*/
#define cudaCheckError() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

/*�������� ��� ��������� ��� ������� � ��� ��� C_h, C ��� ��������� �� ������������ ��� ��������� ���� CPU ��� ���� GPU ����������.*/
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
/*����������� ��� ���������� ������� �.*/
double *transpose(double *A, int �_rows, int �_columns) {
	double *Ah_transpose;
	int size = �_rows * �_columns * sizeof(double);
	Ah_transpose = (double *)malloc(size);
	if ((Ah_transpose == NULL)) {
		printf("Could not allocate memory.\n");
		exit(0);
	}
	int k = 0;
	for (int i = 0; i < �_columns; i++)
		for (int j = 0; j < �_rows; j++) {
			Ah_transpose[k] = A[j*�_columns + i];
			k++;
		}
	return Ah_transpose;
}
/*����������� ��� ��������������� 2 ������� ��� CPU(host), A * B = C.*/
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


__global__ void MatMul(double *A_d, double *C_d, int ARows, int ACols) {

	double CValue = 0.0;
	//����������� ��� ������� ��� ������� ��� ��� ������
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;
	//����� �����������
	int Var0 = blockIdx.y*TILE_DIM + threadIdx.x;
	int Var1 = threadIdx.y*ACols + Var0;
	int Var2 = threadIdx.y*ACols + Col;
	int Var3 = TILE_DIM*ACols;
	int Var4 = ((blockIdx.y * blockDim.y + threadIdx.y)*ACols) + (blockIdx.x*blockDim.x) + threadIdx.x;
	int Var5 = threadIdx.y;
	//����� ������ ������
	__shared__ volatile double As[TILE_DIM][TILE_DIM];
	__shared__ volatile double Bs[TILE_DIM][TILE_DIM];

	int counter = (TILE_DIM + ARows - 1) / TILE_DIM;

	for (int k = 0; k < counter; k++) {
		//���������� ��� ��������� ��� tiles ��� ���������� ����� ����� ��� �������
		if (Var5 < ARows && Var0 < ACols)
			As[threadIdx.x][threadIdx.y] = A_d[k*Var3 + Var1];
		else
			As[threadIdx.x][threadIdx.y] = 0.0;

		if (Var5 < ARows && Col < ACols)
			Bs[threadIdx.y][threadIdx.x] = A_d[k*Var3 + Var2];
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0;
		Var5 = k*TILE_DIM + threadIdx.y;
		__syncthreads();
		//����������� ���������� �����
		CValue = CValue + As[threadIdx.y][0] * Bs[0][threadIdx.x]
			+ As[threadIdx.y][1] * Bs[1][threadIdx.x]
			+ As[threadIdx.y][2] * Bs[2][threadIdx.x]
			+ As[threadIdx.y][3] * Bs[3][threadIdx.x]
			+ As[threadIdx.y][4] * Bs[4][threadIdx.x]
			+ As[threadIdx.y][5] * Bs[5][threadIdx.x]
			+ As[threadIdx.y][6] * Bs[6][threadIdx.x]
			+ As[threadIdx.y][7] * Bs[7][threadIdx.x]
			+ As[threadIdx.y][8] * Bs[8][threadIdx.x]
			+ As[threadIdx.y][9] * Bs[9][threadIdx.x]
			+ As[threadIdx.y][10] * Bs[10][threadIdx.x]
			+ As[threadIdx.y][11] * Bs[11][threadIdx.x]
			+ As[threadIdx.y][12] * Bs[12][threadIdx.x]
			+ As[threadIdx.y][13] * Bs[13][threadIdx.x]
			+ As[threadIdx.y][14] * Bs[14][threadIdx.x]
			+ As[threadIdx.y][15] * Bs[15][threadIdx.x];
		__syncthreads();
	}
	//���������� ������� �����
	if (Row < ACols && Col < ACols)
		C_d[Var4] = CValue;
}


int main() {
	double *A_h, *C_h;	// ������� ��� ����� ���� host
	double *A_d, *C_d; 	// ������� ��� ����� ��� device
	double *At_h;	//������� ��� ����� ���� host ��� ��� ��������� ��� �
	double *C;		//������� ��� ����� ���� host ��� ��� ���������� ��� ��������� ��� CPU
	int N, M;	//���������� ��� ������� � (�*�)
	int size_A, size_C;	//������ ��� ������� ��� �����
	int i, j;	//��������

	//����� ��� cudaEvent API �� ������� ��������
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaCheckError();
	cudaEventCreate(&stop);
	cudaCheckError();
	float milliseconds = 0;

	N = 1024;	//� ������� ������� ��� ������� � 
	M = 1024; 	//� ������� ������ ��� ������� �

	size_A = N * M * sizeof(double);
	size_C = M * M * sizeof(double);

	A_h = (double *)malloc(size_A);
	At_h = (double *)malloc(size_A);
	C_h = (double *)malloc(size_C);
	C = (double *)malloc(size_C);

	if ((A_h == NULL) || (At_h == NULL) || (C_h == NULL) || (C == NULL)) {
		printf("Could not allocate memory.\n");
		exit(0);
	}
	//������������ ��� ������� � �� ������� �����.
	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			A_h[i*M + j] = rand() / (double)RAND_MAX;
		}
	}

	// �������� ������ ��� device ��� �� ����������
	cudaMalloc((void **)&A_d, size_A);
	cudaCheckError();
	//cudaMalloc((void **)&B_d, size);
	cudaMalloc((void **)&C_d, size_C);
	cudaCheckError();

	// ��������� A_h ��� device
	cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
	cudaCheckError();

	//��������������� ���� �� ���� ��� �� ������ ��� block �� ���� ��������
	unsigned int numBlocksX = (M - 1) / BLOCK_SIZE_PER_DIM + 1;
	unsigned int numBlocksY = (M - 1) / BLOCK_SIZE_PER_DIM + 1;
	//������� ���������� ���������
	dim3 dimGrid(numBlocksX, numBlocksY, 1);
	//������� ���������� block
	dim3 dimBlock(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM, 1);

	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	cudaCheckError();
	cudaEventRecord(start);
	cudaCheckError();
	MatMul << <dimGrid, dimBlock >> >(A_d, C_d, N, M);
	cudaCheckError();
	cudaEventRecord(stop);
	cudaCheckError();
	cudaEventSynchronize(stop);
	cudaCheckError();
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaCheckError();
	printf("Time Elapsed:%f \n", milliseconds);

	// ��������� ��� ������������� C_d ���� host
	cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);
	cudaCheckError();

	// ����������� ������ ��� device
	cudaFree(A_d);
	cudaCheckError();
	cudaFree(C_d);
	cudaCheckError();

	//����������� ��� ��������� ��� CPU, �_t * A_h = C.
	At_h = transpose(A_h, N, M);
	C = MatrixMulOnHost(At_h, A_h, N, M);

	//�������� ��� ������� A_h, C_h ��� C.
	//printMatrices(A_h, C_h, C, M, N);

	//���������� ��� ��������� ��� ��������������� ��� GPU.
	for (i = 0; i < M*M; i++) {
		if (C[i] - C_h[i] > 0.000001) {
			printf("The matrix multiplication on GPU was unsuccessful!\n");
			exit(0);
		}
	}
	printf("The matrix multiplication on GPU was successful!\n");
} 