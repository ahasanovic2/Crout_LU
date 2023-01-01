#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <omp.h>
#include <cmath>

using std::endl;
using std::cout;
using namespace Eigen;

const int NUM_OF_TRIES = 3;
const int MATRIX_SIZE = 5000;
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

void printMatrix(double** A) {
    std::cout << "Matrix elements are: " << std::endl;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE - 1; j++) {
            std::cout << A[i][j] << ", ";
        }
        std::cout << A[i][MATRIX_SIZE - 1] << std::endl;
    }
}

// ------------------------ Crout LU decomposition ------------------ //

double** generateMatrixA(int n) {
    double** A = (double**)malloc(sizeof(double*) * n);
    for (int i = 0; i < n; i++) {
        A[i] = (double*)malloc(sizeof(double) * n);
    }

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(1, 1000);


    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = distribution(generator);
        }
    }

    return A;
}

double** generateMatrixLU(int n) {
    double** A = (double**)malloc(sizeof(double*) * n);
    for (int i = 0; i < n; i++) {
        A[i] = (double*)malloc(sizeof(double) * n);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = 0;
        }
    }

    return A;
}

void deallocateMemory(double** A, int n) {
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
}

int crout_sequential(double** A, double** L, double** U, int n) {
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

int crout_parallel(double** A, double** L, double** U, int n) {
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

void crout_sequential_test(double** A, int n) {
    double** L = generateMatrixLU(n);
    double** U = generateMatrixLU(n);

    {
        Timer t("SEQUENTIAL");
        int rj = crout_sequential(A, L, U, n);
        if (rj) {
            std::cout << "Successful!" << std::endl;
        }
        else {
            std::cout << "Matrix is singular!" << std::endl;
        }
    }

    deallocateMemory(L, n);
    deallocateMemory(U, n);
}

void crout_parallel_test(double** A, int n) {
    double** L = generateMatrixLU(n);
    double** U = generateMatrixLU(n);

    {
        Timer t("STATIC CROUT 1");
        crout_parallel(A, L, U, n);
    }

    deallocateMemory(L, n);
    deallocateMemory(U, n);
}

void crout_test(int n) {
    double** A = generateMatrixA(n);

    crout_sequential_test(A, n);

    std::cout << std::endl;

    std::cout << "Maximum number of threads on current device: " << omp_get_max_threads() << std::endl;
    std::cout << "Matrix size: " << n << "x" << n << std::endl;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_OF_THREADS);

    for (int i = 0; i < NUM_OF_TRIES; i++) {

        std::cout << "TRY " << i + 1 << ":" << std::endl;
        crout_parallel_test(A, n);
        std::cout << std::endl;
    }

    omp_set_num_threads(NUM_OF_THREADS * 2);

    for (int i = 0; i < NUM_OF_TRIES; i++) {

        std::cout << "TRY " << i + 1 << ":" << std::endl;
        crout_parallel_test(A, n);
        std::cout << std::endl;
    }

    omp_set_num_threads(NUM_OF_THREADS * 4);

    for (int i = 0; i < NUM_OF_TRIES; i++) {

        std::cout << "TRY " << i + 1 << ":" << std::endl;
        crout_parallel_test(A, n);
        std::cout << std::endl;
    }

    deallocateMemory(A, n);
}

// ------------------------------------------------------------------ //


// ---------------------------- CHOLESKY ---------------------------- //

void makeSymmetric(double** A, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j)
                A[i][j] = A[j][i];
}

bool is_positive_definite(double** A, int n) {
    Eigen::Map<Eigen::MatrixXd> M(A[0], n, n);
    SelfAdjointEigenSolver<Matrix3d> eigensolver(M);
    Vector3d eigenvalues = eigensolver.eigenvalues();
    for (int i = 0; i < n; i++)
        if (eigenvalues[i] <= 0)
            return false;
    return true;
}

void cholesky1_sequential(double** A, double** L, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = 0;
            for (int k = 0; k < j; k++) {
                s += L[i][k] * L[j][k];
            }
            L[i][j] = (i == j) ? sqrt(A[i][i] - s) : (1.0 / L[j][j] * (A[i][j] - s));
        }
    }
}

void cholesky1_parallel(double** A, double** L, int n) {
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        if (i == n-1)
            std::cout << "Number of threads being used: " << omp_get_num_threads() << std::endl;
        for (int j = 0; j <= i; j++) {
            double s = 0;
            for (int k = 0; k < j; k++) {
                s += L[i][k] * L[j][k];
            }
            L[i][j] = (i == j) ? sqrt(A[i][i] - s) : (1.0 / L[j][j] * (A[i][j] - s));
        }
    }
}

void test_cholesky1 (int n) {
    double** A = generateMatrixA(n);
    double** L_S = generateMatrixLU(n);
    double** L_P = generateMatrixLU(n);
    makeSymmetric(A, n);
    {
        Timer t("Check if matrix is positive definite");
        cout << "Matrix is " << (is_positive_definite(A, n) ? "not" : "") << "positive definite" << endl;
    }
    {
        Timer t("CHOLESKY 1 SEQUENTIAL");
        cholesky1_sequential(A, L_S, n);
    }
    {
        Timer t("CHOLESKY 1 PARALLEL");
        cholesky1_parallel(A, L_P, n);
    }
    free(A);
    free(L_S);
    free(L_P);
}

double* generate_matrix(int n) {
    double* A = (double*)calloc(n * n, sizeof(double));
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(1, 1000);


    for (int i = 0; i < n * n; i++) {
        A[i] = distribution(generator);
    }

    return A;
}

void makeSymmetric2(double* A, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j)
                A[j * n + i] = A[i * n + j];
}

bool is_positive_definite2(double* A, int n) {
    Matrix3d M = Map<Matrix3d>(A, n, n);
    SelfAdjointEigenSolver<Matrix3d> eigensolver(M);
    Vector3d eigenvalues = eigensolver.eigenvalues();
    for (int i = 0; i < n; i++)
        if (eigenvalues[i] <= 0)
            return false;
    return true;
}

double* cholesky2_parallel(double* A, int n) {
    double* L = (double*)calloc(n * n, sizeof(double));
    if (L == NULL)
        exit(EXIT_FAILURE);

    for (int j = 0; j < n; j++) {
        double s = 0;
        for (int k = 0; k < j; k++) {
            s += L[j * n + k] * L[j * n + k];
        }
        L[j * n + j] = sqrt(A[j * n + j] - s);
#pragma omp parallel for num_threads(4)
        for (int i = j + 1; i < n; i++) {
            double s = 0;
            for (int k = 0; k < j; k++) {
                s += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (1.0 / L[j * n + j] * (A[i * n + j] - s));
        }
    }
    return L;
}

double* cholesky2_sequential(double* A, int n) {
    double* L = (double*)calloc(n * n, sizeof(double));
    if (L == NULL)
        exit(EXIT_FAILURE);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < (i + 1); j++) {
            double s = 0;
            for (int k = 0; k < j; k++) {
                s += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (i == j) ? sqrt(A[i * n + i] - s) : (1.0 / L[j * n + j] * (A[i * n + j] - s));
        }
    }
    return L;
}

void test_cholesky2(int n) {
    double* A = generate_matrix(n);
    makeSymmetric2(A, n);
    {
        Timer t("Check if matrix is positive definite");
        cout << "Matrix is " << (is_positive_definite2(A, n) ? "not" : "") << "positive definite" << endl;
    }
    {
        Timer t("CHOLESKY 2 SEQUENTIAL");
        auto l = cholesky2_sequential(A, n);
        free(l);
    }
    {
        Timer t("CHOLESKY 2 PARALLEL");
        auto l = cholesky2_parallel(A, n);
        free(l);
    }
    free(A);
}
// ------------------------------------------------------------------ //

int main()
{
    cout << "Insert number of rows for matrix: ";
    int broj;
    std::cin >> broj;
    test_cholesky1(broj);
    //test_cholesky2(broj);
    //crout_test(broj);
    return 0;

}