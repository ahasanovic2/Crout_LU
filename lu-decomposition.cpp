#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <omp.h>

const int matrix_size = 1500;
const int num_threads = 8;
const int num_of_tries = 5;

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

// Algorithms for LU decomposition, different variations

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

void crout_2(double** A, double** L, double** U, int n) {
    int i, j, k;
    double sum = 0;
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }
    for (j = 0; j < n; j++) {
        //i=j case
        sum = 0;
        for (k = 0; k < j; k++) {
            sum = sum + L[j][k] * U[k][j];
        }
        L[j][j] = A[j][j] - sum;
        int x = j + (n - j) / 2;
#pragma omp parallel sections
        {
#pragma omp section
            {
                for (int i = j + 1; i < x; i++) {
                    double sum = 0;
                    for (int k = 0; k < j; k++) {
                        sum = sum + L[i][k] * U[k][j];
                    }
                    L[i][j] = A[i][j] - sum;
                }
            }
#pragma omp section
            {
                for (int i = x; i < n; i++) {
                    double sum = 0;
                    for (int k = 0; k < j; k++) {
                        sum = sum + L[i][k] * U[k][j];
                    }
                    L[i][j] = A[i][j] - sum;
                }

            }
#pragma omp section
            {
                for (int i = j; i < x; i++) {
                    double sum = 0;
                    for (int k = 0; k < j; k++) {
                        sum = sum + L[j][k] * U[k][i];
                    }
                    if (L[j][j] == 0) {
                        exit(0);
                    }
                    U[j][i] = (A[j][i] - sum) / L[j][j];
                }
            }
#pragma omp section
            {
                for (int i = x; i < n; i++) {
                    double sum = 0;
                    for (int k = 0; k < j; k++) {
                        sum = sum + L[j][k] * U[k][i];
                    }
                    if (L[j][j] == 0) {
                        exit(0);
                    }
                    U[j][i] = (A[j][i] - sum) / L[j][j];
                }
            }
        }
    }

}

void crout_4(double** m, double** l, double** u, int size) {
    int i = 0, j = 0, k = 0;
    double factor;

#pragma omp for 

    for (j = 0; j < size - 1; j++) {
        for (i = j + 1; i < size; i++) {
            factor = m[i][j] / m[j][j];
            for (k = 0; k < size; k++) {
                u[i][k] = m[i][k] - (m[j][k] * factor);
            }
            l[i][j] = factor;
        }
        //matrix copy m = u
        for (i = 0; i < size; i++) {
            for (k = 0; k < size; k++) {
                m[i][k] = u[i][k];
            }
        }
        //copy end
    }
    //freeMatrix(mat);
    for (i = 0; i < size; i++) {
        l[i][i] = 1;
    }
}

void crout_6(double** a, double** l, double** u, int size, int chunkSize = 16) {
#pragma omp parallel shared(a,l,u)
    {
        for (int i = 0; i < size; i++)
        {
            //for each row....
            //rows are split into seperate threads for processing
#pragma omp for schedule(dynamic, chunkSize)
            for (int j = 0; j < size; j++)
            {
                //if j is smaller than i, set l[j][i] to 0
                if (j < i)
                {
                    l[j][i] = 0;
                    continue;
                }
                //otherwise, do some math to get the right value
                l[j][i] = a[j][i];
                for (int k = 0; k < i; k++)
                {
                    //deduct from the current l cell the value of these 2 values multiplied
                    l[j][i] = l[j][i] - l[j][k] * u[k][i];
                }
            }
            //for each row...
            //rows are split into seperate threads for processing
#pragma omp for schedule(dynamic, chunkSize)
            for (int j = 0; j < size; j++)
            {
                //if j is smaller than i, set u's current index to 0
                if (j < i)
                {
                    u[i][j] = 0;
                    continue;
                }
                //if they're equal, set u's current index to 1
                if (j == i)
                {
                    u[i][j] = 1;
                    continue;
                }
                //otherwise, do some math to get the right value
                u[i][j] = a[i][j] / l[i][i];
                for (int k = 0; k < i; k++)
                {
                    u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
                }
            }
        }
    }
}


// Testing methods

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

void testCroutParallel2(double** A, int n) {
    double** L = generateMatrixLU();
    double** U = generateMatrixLU();

    {
        Timer t("PARALLEL 2");
        crout_2(A, L, U, n);
    }

    deallocateMemory(L);
    deallocateMemory(U);
}

void testCroutParallel4(double** A, int n) {
    double** L = generateMatrixLU();
    double** U = generateMatrixLU();

    {
        Timer t("PARALLEL 4");
        crout_4(A, L, U, n);
    }

    deallocateMemory(L);
    deallocateMemory(U);
}

void testCroutParallel6(double** A, int n) {
    double** L = generateMatrixLU();
    double** U = generateMatrixLU();

    {
        Timer t("PARALLEL 6");
        crout_6(A, L, U, n);
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
        //testCroutParallel2(A, matrix_size);

        std::cout << std::endl;
    }
    deallocateMemory(A);
    return 0;
    
}
