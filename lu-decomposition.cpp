#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <omp.h>

const int matrix_size = 2000;
const int num_threads = 8;
const int num_of_tries = 15;

class Timer {
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1, 1000000>> second_;
    std::chrono::time_point<clock_> beg_;
    const char* header;
public:
    Timer(const char* header = "") : beg_(clock_::now()), header(header) {}
    ~Timer() {
        double e = elapsed();
        std::cout << header << ": " << e / 1000000 << " seconds" << std::endl;
    }
    void reset() {
        beg_ = clock_::now();
    }
    double elapsed() const {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }
};

// Helper functions that are used for clean code

double** generateMatrixA() {
    double** A = (double**)malloc(sizeof(double*) * matrix_size);
    for (int i = 0; i < matrix_size; i++) {
        A[i] = (double*)malloc(sizeof(double) * matrix_size);
    }

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(1, 1000);


    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            A[i][j] = distribution(generator);
        }
    }

    return A;
}

double** generateMatrixLU() {
    double** A = (double**)malloc(sizeof(double*) * matrix_size);
    for (int i = 0; i < matrix_size; i++) {
        A[i] = (double*)malloc(sizeof(double) * matrix_size);
    }

    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            A[i][j] = 0;
        }
    }

    return A;
}

void deallocateMemory(double** A) {
    for (int i = 0; i < matrix_size; i++) {
        free(A[i]);
    }
    free(A);
}

void printMatrix(double** A) {
    std::cout << "Matrix elements are: " << std::endl;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size-1; j++) {
            std::cout << A[i][j] << ", ";
        }
        std::cout << A[i][matrix_size - 1] << std::endl;
    }
}

/*-----------------------------
Crout algorithm for LU decomposition
-----------------------------*/
// Sequential version of Crout algorithm
void crout_0(double** A, double** L, double** U, int n) {
    int i, j, k;
    double sum = 0;
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }
    for (j = 0; j < n; j++) {
        for (i = j; i < n; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[i][k] * U[k][j];
            }
            L[i][j] = A[i][j] - sum;
        }
        for (i = j; i < n; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[j][k] * U[k][i];
            }
            if (L[j][j] == 0) {
                exit(0);
            }
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
}

// Parallel version of Crout algorithm
void crout_1(double** A, double** L, double** U, int n) {
    int i, j, k;
    double sum = 0;
#pragma omp parallel for
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }
#pragma omp parallel for private(i,j,k,sum) schedule(dynamic, 8)
    for (j = 0; j < n; j++) {
        if(j == n-1) 
            std::cout << "Number of threads being used: " << omp_get_num_threads() << std::endl;
        //#pragma omp parallel for private(i,k,sum)
        for (i = j; i < n; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[i][k] * U[k][j];
            }
            L[i][j] = A[i][j] - sum;
        }
        //#pragma omp parallel for private(i,k,sum)
        for (i = j; i < n; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[j][k] * U[k][i];
            }
            if (L[j][j] == 0) {
                exit(0);
            }
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
    
}

/*-----------------------------
Void methods used for testing sequential and parallel Crout algorithm
-----------------------------*/

void testCroutSequencial(double** A, int n) {
    double** L = generateMatrixLU();
    double** U = generateMatrixLU();

    {
        Timer t("SEQUENTIAL");
        crout_0(A, L, U, n);
    }
    
    deallocateMemory(L);
    deallocateMemory(U);
}

void testCroutParallel1(double** A, int n) {
    double** L = generateMatrixLU();
    double** U = generateMatrixLU();

    {
        Timer t("PARALLEL 1");
        crout_1(A, L, U, n);
    }

    deallocateMemory(L);
    deallocateMemory(U);
}

int main()
{
    double** A = generateMatrixA();

    testCroutSequencial(A, matrix_size);

    std::cout << std::endl;

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    
    std::cout << "Maximum number of threads on current device: " << omp_get_max_threads() << std::endl;
    std::cout << "Matrix size: " << matrix_size << "x" << matrix_size << std::endl;

    for (int i = 0; i < num_of_tries; i++) {

        std::cout << "TRY " << i + 1 << ":" << std::endl;
        testCroutParallel1(A, matrix_size);
        std::cout << std::endl;
    }
    deallocateMemory(A);
    return 0;
    
}