#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <omp.h>

const int NUM_OF_TRIES = 3;
const int MATRIX_SIZE = 250;
const int NUM_OF_THREADS = 2;

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
    double** A = (double**)malloc(sizeof(double*) * MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        A[i] = (double*)malloc(sizeof(double) * MATRIX_SIZE);
    }

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(1, 1000);


    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            A[i][j] = distribution(generator);
        }
    }

    return A;
}

double** generateMatrixLU() {
    double** A = (double**)malloc(sizeof(double*) * MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        A[i] = (double*)malloc(sizeof(double) * MATRIX_SIZE);
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            A[i][j] = 0;
        }
    }

    return A;
}

void deallocateMemory(double** A) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        free(A[i]);
    }
    free(A);
}

void printMatrix(double** A) {
    std::cout << "Matrix elements are: " << std::endl;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE - 1; j++) {
            std::cout << A[i][j] << ", ";
        }
        std::cout << A[i][MATRIX_SIZE - 1] << std::endl;
    }
}

/*-----------------------------
Crout algorithm for LU decomposition
-----------------------------*/
// Sequential version of Crout algorithm
int crout_0(double** A, double** L, double** U, int n) {
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
                return 0;
            }
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
    return 1;
}

// Parallel version of Crout algorithm
int crout_1_runtime(double** A, double** L, double** U, int n) {
    int i, j, k;
    double sum = 0;
#pragma omp parallel for
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }
#pragma omp parallel for private(i,j,k,sum) schedule(runtime)
    for (j = 0; j < n; j++) {
        if (j == n - 1)
            std::cout << "Number of threads being used: " << omp_get_num_threads() << std::endl;
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
                std::cout << "Matrix is singular! " << std::endl;
                exit(0);
            }
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
    return 1;
}

int crout_1_static(double** A, double** L, double** U, int n) {
    int i, j, k;
    double sum = 0;
#pragma omp parallel for
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }
#pragma omp parallel for private(i,j,k,sum) schedule(static)
    for (j = 0; j < n; j++) {
        if (j == n - 1)
            std::cout << "Number of threads being used: " << omp_get_num_threads() << std::endl;
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
                std::cout << "Matrix is singular! " << std::endl;
                exit(0);
            }
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
    return 1;
}

/*-----------------------------
Partial pivoting algorithm for LU decomposition
-----------------------------*/

static double matrix[MATRIX_SIZE][MATRIX_SIZE];
static double matrix_p[MATRIX_SIZE][MATRIX_SIZE];
const double epsilon = 1e-5;
static int perm[MATRIX_SIZE];
static int perm_p[MATRIX_SIZE];

// Sequential version of Partial pivoting algorithm
void partial_pivoting() {
    int i, j, k, pindex;
    double* lu = &matrix[0][0];
    double* row = lu;
    double* pivot_row = nullptr;
    double max;

    // for each row and column, k = 0, ..., n-1
    for (k = 0; k < MATRIX_SIZE; ++k) {
        pindex = k;
        max = fabs(*(row + k));
        {
            // find the pivot row
            for (j = k + 1; j < MATRIX_SIZE; ++j) {
                double temp = fabs(lu[j * MATRIX_SIZE + k]);
                if (max < temp) {
                    max = temp;
                    pindex = j;
                    pivot_row = &lu[j * MATRIX_SIZE];
                }
            }
            // and if the pivot row differs from the current row
            // then interchange the two rows
            if (pindex != k)
                for (j = 0; j < MATRIX_SIZE; ++j)
                    std::swap(*(row + j), *(pivot_row + j));

            // and if the matrix is singular, return error
            if (fabs(*(row + k)) < epsilon)
                throw std::runtime_error("Matrix is singular");

            // otherwise find the upper triangular matrix elements for row k
            for (j = k + 1; j < MATRIX_SIZE; ++j)
                *(row + j) /= *(row + k);

            // update remaining matrix
            for (i = k + 1; i < MATRIX_SIZE; ++i)
                for (j = k + 1; j < MATRIX_SIZE; ++j)
                    lu[i * MATRIX_SIZE + j] -= lu[i * MATRIX_SIZE + k] * *(row + j);
        }
        std::swap(perm[pindex], perm[k]);
        row += MATRIX_SIZE;
    }
}

// Parallel version of Partial pivoting algorithm
void partial_pivoting_parallel() {
    int i, j, k, pindex;
    double* lu = &matrix_p[0][0];
    double* row = lu;
    double* pivot_row;
    double max;

    // for each row and column, k = 0, ..., n-1
    for (k = 0; k < MATRIX_SIZE; ++k) {

        pindex = k;
        max = fabs(*(row + k));
#pragma omp parallel num_threads(NUM_OF_THREADS), default(none), private(i, j), shared(k, lu, row, perm_p, pindex, pivot_row, max)
        {
            // find the pivot row
#pragma omp for
            for (j = k + 1; j < MATRIX_SIZE; ++j) {
                double temp = fabs(lu[j * MATRIX_SIZE + k]);
#pragma omp critical
                if (max < temp) {
                    max = temp;
                    pindex = j;
                    pivot_row = &lu[j * MATRIX_SIZE];
                }
            }

            // and if the pivot row differs from the current row
            // then interchange the two rows
            if (pindex != k)
#pragma omp for
                for (j = 0; j < MATRIX_SIZE; ++j)
                    std::swap(*(row + j), *(pivot_row + j));

            // and if the matrix is singular, return error
            if (fabs(*(row + k)) < epsilon)
                throw std::runtime_error("Matrix is singular");

            // otherwise find the upper triangular matrix elements for row k
#pragma omp for
            for (j = k + 1; j < MATRIX_SIZE; ++j)
                *(row + j) /= *(row + k);

            // update remaining matrix
#pragma omp for
            for (i = k + 1; i < MATRIX_SIZE; ++i)
                for (j = k + 1; j < MATRIX_SIZE; ++j)
                    lu[i * MATRIX_SIZE + j] -= lu[i * MATRIX_SIZE + k] * *(row + j);
        }

        std::swap(perm_p[pindex], perm_p[k]);
        row += MATRIX_SIZE;
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
        int rj = crout_0(A, L, U, n);
        if (rj) {
            std::cout << "Successful!" << std::endl;
        }
        else {
            std::cout << "Matrix is singular!" << std::endl;
        }
    }

    deallocateMemory(L);
    deallocateMemory(U);
}

void testCroutParallel_runtime(double** A, int n) {
    double** L = generateMatrixLU();
    double** U = generateMatrixLU();

    {
        Timer t("RUNTIME CROUT 1");
        crout_1_runtime(A, L, U, n);
    }

    deallocateMemory(L);
    deallocateMemory(U);
}

void testCroutParallel_dynamic(double** A, int n) {
    double** L = generateMatrixLU();
    double** U = generateMatrixLU();

    {
        Timer t("DYNAMIC CROUT 1");
        crout_1_runtime(A, L, U, n);
    }

    deallocateMemory(L);
    deallocateMemory(U);
}

void testCroutParallel1_static(double** A, int n) {
    double** L = generateMatrixLU();
    double** U = generateMatrixLU();

    {
        Timer t("PARALLEL 1 with static scheduling");
        crout_1_static(A, L, U, n);
    }

    deallocateMemory(L);
    deallocateMemory(U);
}

/*-----------------------------
Void methods used for testing sequential and parallel Partial pivoting algorithm
-----------------------------*/

void fill_matrices() {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(1, 1000);


    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = matrix_p[i][j] = distribution(generator);
        }
    }
}

void test_partial_pivoting_sequential() {
    {
        Timer t("Partial Pivoting Sequential");
        try {
            partial_pivoting();
        }
        catch (std::exception e) {
            std::cout << e.what() << std::endl;
        }
    }
}

void test_partial_pivoting_parallel() {
    {
        Timer t("Partial Pivoting Parallel");
        try {
            partial_pivoting_parallel();
        }
        catch (std::exception e) {
            std::cout << e.what() << std::endl;
        }
    }
}

/*-----------------------------
Separate methods for testing Crout and Partial pivoting algorithm
-----------------------------*/

void test_crout_runtime() {
    double** A = generateMatrixA();

    testCroutSequencial(A, MATRIX_SIZE);

    std::cout << std::endl;

    std::cout << "Maximum number of threads on current device: " << omp_get_max_threads() << std::endl;
    std::cout << "Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_OF_THREADS);

    for (int i = 0; i < NUM_OF_TRIES; i++) {

        std::cout << "TRY " << i + 1 << ":" << std::endl;
        testCroutParallel_runtime(A, MATRIX_SIZE);
        std::cout << std::endl;
    }
    
    omp_set_num_threads(NUM_OF_THREADS*2);

    for (int i = 0; i < NUM_OF_TRIES; i++) {

        std::cout << "TRY " << i + 1 << ":" << std::endl;
        testCroutParallel_runtime(A, MATRIX_SIZE);
        std::cout << std::endl;
    }
    
    omp_set_num_threads(NUM_OF_THREADS*4);

    for (int i = 0; i < NUM_OF_TRIES; i++) {

        std::cout << "TRY " << i + 1 << ":" << std::endl;
        testCroutParallel_runtime(A, MATRIX_SIZE);
        std::cout << std::endl;
    }

    deallocateMemory(A);
}

void test_crout_dynamic() {
    double** A = generateMatrixA();

    testCroutSequencial(A, MATRIX_SIZE);

    std::cout << std::endl;

    std::cout << "Maximum number of threads on current device: " << omp_get_max_threads() << std::endl;
    std::cout << "Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_OF_THREADS);

    for (int i = 0; i < NUM_OF_TRIES; i++) {

        std::cout << "TRY " << i + 1 << ":" << std::endl;
        testCroutParallel_dynamic(A, MATRIX_SIZE);
        std::cout << std::endl;
    }

    omp_set_num_threads(NUM_OF_THREADS * 2);

    for (int i = 0; i < NUM_OF_TRIES; i++) {

        std::cout << "TRY " << i + 1 << ":" << std::endl;
        testCroutParallel_dynamic(A, MATRIX_SIZE);
        std::cout << std::endl;
    }

    omp_set_num_threads(NUM_OF_THREADS * 4);

    for (int i = 0; i < NUM_OF_TRIES; i++) {

        std::cout << "TRY " << i + 1 << ":" << std::endl;
        testCroutParallel_dynamic(A, MATRIX_SIZE);
        std::cout << std::endl;
    }

    deallocateMemory(A);
}

void test_pivot() {
    fill_matrices();
    test_partial_pivoting_sequential();
    test_partial_pivoting_parallel();
}

int main()
{
    //test_pivot();
    std::cout << "RUNTIME SCHEDULING: " << std::endl;
    test_crout_runtime();
    std::cout << "DYNAMIC SCHEDULING: " << std::endl;
    test_crout_dynamic();
    return 0;

}