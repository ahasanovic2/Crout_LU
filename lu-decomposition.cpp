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

void printMatrix(double** A, int n) {
    std::cout << "Matrix elements are: " << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - 1; j++) {
            std::cout << A[i][j] << ", ";
        }
        std::cout << A[i][n - 1] << std::endl;
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
    //Eigen::Map<Eigen::MatrixXd> M(A[0], n, n);
    //SelfAdjointEigenSolver<Matrix3d> eig(M);
    //Vector3d eigenvalues = eig.eigenvalues();
    //cout << "Eigenvalues are: ";
    ///*for (int i = 0; i < n; i++)
    //    cout << eigenvalues[i] << ", ";
    //cout << endl;*/
    //for (int i = 0; i < n; i++)
    //    if (eigenvalues[i] <= 0)
    //        return false;
    //return true;

    double* eigenvalues = (double*)calloc(n, sizeof(double));
    double* P = (double*)calloc(n+1, sizeof(double));
    P[0] = -1;
    for (int i = 0; i < n; i++) {
        P[i + 1] = 0;
        for (int j = 0; j < n; j++) {
            P[i + 1] -= A[j][i] * A[j][i];
        }
    }
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = 0;
        for (int j = 0; j < i + 1; j++) {
            eigenvalues[i] += P[j] * pow(eigenvalues[i], i - j);
        }
    }
    for (int i = 0; i < n; i++)
        cout << eigenvalues[i] << ", ";
    bool povrat = true;
    for (int i = 0; i < n; i++)
        if (eigenvalues[i] <= 0)
            povrat = false;
    
    free(eigenvalues);
    free(P);
    return povrat;
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

// ------------------------------ LDLT ------------------------------ //

double* generate_help_array(int n) {
    double* l = (double*)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
        l[i] = 0;
    return l;
}

double* ldlt_sequential(double** A, const int n) {
    double* l = generate_help_array(n);
    for (int i = 0; i < n; i++)
    {
        // Get the ith column of the matrix
        double* a = generate_help_array(n);
        for (int j = 0; j < n; j++)
            a[j] = A[j][i];

        // Compute the ith element of the diagonal of L
        double sum = 0;
        for (int j = 0; j < i; j++)
            sum += l[j] * a[j] * a[j];
        l[i] = a[i] - sum;

        // Update the rest of the column
        for (int j = i + 1; j < n; j++)
            a[j] = (a[j] - (A[j][i] - sum)) / l[i];

        // Update the ith row and column of the matrix
        for (int j = 0; j < n; j++)
        {
            A[j][i] = a[j];
            A[i][j] = a[j];
        }
        free(a);
    }
    return l;
}

double* ldlt_parallel(double** A, const int n) {
    double* l = (double*)calloc(n, sizeof(double));
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < n; i++)
        {
            // Get the ith column of the matrix
            double* a = (double*)calloc(n, sizeof(double));
            for (int j = 0; j < n; j++)
                a[j] = A[j][i];

            // Compute the ith element of the diagonal of L
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += l[j] * a[j] * a[j];
            l[i] = a[i] - sum;

            // Update the rest of the column
            for (int j = i + 1; j < n; j++)
                a[j] = (a[j] - (A[j][i] - sum)) / l[i];

            // Update the ith row and column of the matrix
            for (int j = 0; j < n; j++)
            {
                A[j][i] = a[j];
                A[i][j] = a[j];
            }
            free(a);
        }
    }
    return l;
}

void test_ldlt1(int n) {
    double** A = generateMatrixA(n);
    makeSymmetric(A, n);
    printMatrix(A, 10);
    {
        Timer t("LDLT SEQUENTIAL");
        auto l = ldlt_sequential(A, n);
        free(l);
    }
    printMatrix(A, 10);
    /*{
        Timer t("LDLT PARALLEL");
        auto l = ldlt_parallel(A, n);
        free(l);
    }*/
    free(A);
}

// ------------------------------------------------------------------ //

void test() {
    int x = 3;
    double** A = (double**)malloc(sizeof(double*) * x);
    for (int i = 0; i < x; i++) {
        A[i] = (double*)malloc(sizeof(double) * x);
    }
    A[0][0] = 1;
    A[0][1] = 2;
    A[0][2] = 3;
    A[1][0] = 4;
    A[1][1] = 5;
    A[1][2] = 6;
    A[2][0] = 7;
    A[2][1] = 8;
    A[2][2] = 9;

    double** L = (double**)malloc(sizeof(double*) * x);
    for (int i = 0; i < x; i++) {
        L[i] = (double*)malloc(sizeof(double) * x);
    }

    L[0][0] = 0;
    L[0][1] = 0;
    L[0][2] = 0;
    L[1][0] = 0;
    L[1][1] = 0;
    L[1][2] = 0;
    L[2][0] = 0;
    L[2][1] = 0;
    L[2][2] = 0;

    printMatrix(A, x);
    cout << "Matrix is " << (is_positive_definite(A, x) ? "" : "not") << " positive definite" << endl;
    cholesky1_sequential(A, L, x);
    printMatrix(L, x);
    deallocateMemory(A,x);
    deallocateMemory(L,x);
}

void test2() {
    // Define a 3x3 matrix
    //double A[3][3] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };

    Eigen::Matrix<double, 3, 3> A; // declare a real (double) 2x2 matrix
    A << 25,15,-5,15,18,0,-5,0,11; // defined the matrix A

    Eigen::EigenSolver<Eigen::Matrix<double, 3, 3> > s(A); // the instance s(A) includes the eigensystem
    std::cout << A << std::endl;
    std::cout << "eigenvalues:" << std::endl;
    std::cout << s.eigenvalues()(0) << std::endl;
    std::cout << s.eigenvalues()(1) << std::endl;
    std::cout << s.eigenvalues()(2) << std::endl;
    std::cout << s.eigenvalues()(2).real() << std::endl;
    std::cout << "eigenvectors=" << std::endl;
    std::cout << s.eigenvectors() << std::endl;
}

int main()
{
    //cout << "Insert number of rows for matrix: ";
    //int broj;
    //std::cin >> broj;
    //test_cholesky1(broj);
    //test_cholesky2(broj);
    //crout_test(broj);
    //test_ldlt1(broj);

    //test();
    test2();
    return 0;

}