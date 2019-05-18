#include "gmres.hpp"
#include "conjugate_gradient.hpp"

#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "gtest/gtest.h"

namespace ublas=boost::numeric::ublas;

TEST(TestGMRES, DenseMatrix)
{
    using V = ublas::vector<double>;
    using M = ublas::matrix<double>;

    V b(3);
    b(0) = 14.0;
    b(1) = 32.0;
    b(2) = 50.0;

    M A(3, 3);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;
    A(2, 0) = 7.0;
    A(2, 1) = 8.0;
    A(2, 2) = 9.0;

    V x(3, 0.0);
    V sol(3, 0.0);
    sol(0) = 1.0;
    sol(1) = 2.0;
    sol(2) = 3.0;

    ublas::gmres<M> solver(A);

    solver(b, x);

    EXPECT_NEAR(ublas::norm_2(x - sol), 0.0, 1e-5);
}

TEST(TestGMRES, Functor)
{
    using V = ublas::vector<double>;
    using M = ublas::matrix<double>;

    V b(3);
    b(0) = 14.0;
    b(1) = 32.0;
    b(2) = 50.0;

    M A(3, 3);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;
    A(2, 0) = 7.0;
    A(2, 1) = 8.0;
    A(2, 2) = 9.0;

    V x(3, 0.0);
    V sol(3, 0.0);
    sol(0) = 1.0;
    sol(1) = 2.0;
    sol(2) = 3.0;

    auto Op = [&A](const auto& v){return ublas::prod(A, v);};
    auto PInv = [](const auto& v)->decltype(auto){return v;};

    ublas::gmres<decltype(Op), decltype(PInv)> solver(Op, PInv);

    solver(b, x);

    EXPECT_NEAR(ublas::norm_2(x - sol), 0.0, 1e-5);
}

TEST(TestGMRES, Vandermonde1)
{
    using V = ublas::vector<double>;
    using M = ublas::matrix<double>;

    V b(3);
    b(0) = -6.0;
    b(1) = 2.0;
    b(2) = 12.0;

    M A(3, 3);
    A(0, 0) = 1.0;
    A(0, 1) = 1.0;
    A(0, 2) = 1.0;
    A(1, 0) = 4.0;
    A(1, 1) = 2.0;
    A(1, 2) = 1.0;
    A(2, 0) = 16.0;
    A(2, 1) = 4.0;
    A(2, 2) = 1.0;

    V x(3, 0.0);
    V sol(3, 0.0);
    sol(0) = -1.0;
    sol(1) = 11.0;
    sol(2) = -16.0;

    ublas::gmres<M> solver(A);

    solver(b, x);

    EXPECT_NEAR(ublas::norm_2(x - sol), 0.0, 1e-5);
}

TEST(TestGMRES, Vandermonde2)
{
    using V = ublas::vector<double>;
    using M = ublas::matrix<double>;

    V b(4);
    b(0) = -6.0;
    b(1) = 2.0;
    b(2) = 12.0;
    b(3) = -10.0;

    M A(4, 4);
    A(0, 0) = 1.0;
    A(0, 1) = 1.0;
    A(0, 2) = 1.0;
    A(0, 3) = 1.0;
    A(1, 0) = 8.0;
    A(1, 1) = 4.0;
    A(1, 2) = 2.0;
    A(1, 3) = 1.0;
    A(2, 0) = 64.0;
    A(2, 1) = 16.0;
    A(2, 2) = 4.0;
    A(2, 3) = 1.0;
    A(3, 0) = 27.0;
    A(3, 1) = 9.0;
    A(3, 2) = 3.0;
    A(3, 3) = 1.0;

    V x(4, 0.0);
    V sol(4, 0.0);
    sol(0) = 9.0;
    sol(1) = -64.0;
    sol(2) = 137.0;
    sol(3) = -88.0;

    ublas::gmres<M> solver(A);

    solver(b, x);

    EXPECT_NEAR(ublas::norm_2(x - sol), 0.0, 1e-6);
}

TEST(TestGMRES, Laplacian)
{
    using V = ublas::vector<double>;
    using M = ublas::compressed_matrix<double>;

    M A(6, 6);
    A(0, 0) = 4.0;
    A(0, 1) = -1.0;
    A(0, 3) = -1.0;
    A(1, 0) = -1.0;
    A(1, 1) = 4.0;
    A(1, 2) = -1.0;
    A(1, 4) = -1.0;
    A(2, 1) = -1.0;
    A(2, 2) = 4.0;
    A(2, 5) = -1.0;
    A(3, 0) = -1.0;
    A(3, 3) = 4.0;
    A(3, 4) = -1.0;
    A(4, 1) = -1.0;
    A(4, 3) = -1.0;
    A(4, 4) = 4.0;
    A(4, 5) = -1.0;
    A(5, 2) = -1.0;
    A(5, 4) = -1.0;
    A(5, 5) = 4.0;

    V sol(6, 0.0);
    sol(0) = 1.0;
    sol(1) = 2.0;
    sol(2) = 3.0;
    sol(3) = 4.0;
    sol(4) = 5.0;
    sol(5) = 6.0;

    V x(6, 0.0);
    V b = ublas::prod(A, sol);

    ublas::gmres<M> solver(A);

    solver(b, x);

    EXPECT_NEAR(ublas::norm_2(x - sol), 0.0, 1e-6);
}

TEST(TestGMRES, JacobiPrecond)
{
    using V = ublas::vector<double>;
    using M = ublas::compressed_matrix<double>;
    using P = ublas::jacobi_precond<M>;

    M A(6, 6, 0.0);
    A(0, 0) = 4.0;
    A(0, 1) = -1.0;
    A(0, 3) = -1.0;
    A(1, 0) = -1.0;
    A(1, 1) = 4.0;
    A(1, 2) = -1.0;
    A(1, 4) = -1.0;
    A(2, 1) = -1.0;
    A(2, 2) = 4.0;
    A(2, 5) = -1.0;
    A(3, 0) = -1.0;
    A(3, 3) = 4.0;
    A(3, 4) = -1.0;
    A(4, 1) = -1.0;
    A(4, 3) = -1.0;
    A(4, 4) = 4.0;
    A(4, 5) = -1.0;
    A(5, 2) = -1.0;
    A(5, 4) = -1.0;
    A(5, 5) = 4.0;

    V sol(6, 0.0);
    sol(0) = 1.0;
    sol(1) = 2.0;
    sol(2) = 3.0;
    sol(3) = 4.0;
    sol(4) = 5.0;
    sol(5) = 6.0;

    V x(6, 0.0);
    V b = ublas::prod(A, sol);

    ublas::gmres<M, P> solver(A);

    solver(b, x);

    EXPECT_NEAR(ublas::norm_2(x - sol), 0.0, 1e-6);
}

TEST(TestCG, DenseMatrix)
{
    using V = ublas::vector<double>;
    using M = ublas::matrix<double>;

    V b(3);
    b(0) = 0.0;
    b(1) = 0.0;
    b(2) = -4.0;

    M A(3, 3);
    A(0, 0) = -2.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.0;
    A(1, 0) = 1.0;
    A(1, 1) = -2.0;
    A(1, 2) = 1.0;
    A(2, 0) = 0.0;
    A(2, 1) = 1.0;
    A(2, 2) = -2.0;

    V x(3, 0.0);
    V sol(3, 0.0);
    sol(0) = 1.0;
    sol(1) = 2.0;
    sol(2) = 3.0;

    ublas::conjugate_gradient<M> solver(A);

    solver(b, x);

    EXPECT_NEAR(ublas::norm_2(x - sol), 0.0, 1e-6);
}

TEST(TestCG, Functor)
{
    using V = ublas::vector<double>;
    using M = ublas::matrix<double>;

    V b(3);
    b(0) = 0.0;
    b(1) = 0.0;
    b(2) = -4.0;

    M A(3, 3);
    A(0, 0) = -2.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.0;
    A(1, 0) = 1.0;
    A(1, 1) = -2.0;
    A(1, 2) = 1.0;
    A(2, 0) = 0.0;
    A(2, 1) = 1.0;
    A(2, 2) = -2.0;

    V x(3, 0.0);
    V sol(3, 0.0);
    sol(0) = 1.0;
    sol(1) = 2.0;
    sol(2) = 3.0;

    auto Op = [&A](const auto& v){return ublas::prod(A, v);};
    auto PInv = [](const auto& v)->decltype(auto){return v;};
    ublas::conjugate_gradient<decltype(Op), decltype(PInv)> solver(Op, PInv);

    solver(b, x);

    EXPECT_NEAR(ublas::norm_2(x - sol), 0.0, 1e-6);
}

TEST(TestCG, Laplacian)
{
    using V = ublas::vector<double>;
    using M = ublas::compressed_matrix<double>;

    M A(6, 6);
    A(0, 0) = 4.0;
    A(0, 1) = -1.0;
    A(0, 3) = -1.0;
    A(1, 0) = -1.0;
    A(1, 1) = 4.0;
    A(1, 2) = -1.0;
    A(1, 4) = -1.0;
    A(2, 1) = -1.0;
    A(2, 2) = 4.0;
    A(2, 5) = -1.0;
    A(3, 0) = -1.0;
    A(3, 3) = 4.0;
    A(3, 4) = -1.0;
    A(4, 1) = -1.0;
    A(4, 3) = -1.0;
    A(4, 4) = 4.0;
    A(4, 5) = -1.0;
    A(5, 2) = -1.0;
    A(5, 4) = -1.0;
    A(5, 5) = 4.0;

    V sol(6, 0.0);
    sol(0) = 1.0;
    sol(1) = 2.0;
    sol(2) = 3.0;
    sol(3) = 4.0;
    sol(4) = 5.0;
    sol(5) = 6.0;

    V x(6, 0.0);
    V b = ublas::prod(A, sol);

    ublas::conjugate_gradient<M> solver(A);

    solver(b, x);

    EXPECT_NEAR(ublas::norm_2(x - sol), 0.0, 1e-6);
}

TEST(TestCG, JacobiPrecond)
{
    using V = ublas::vector<double>;
    using M = ublas::compressed_matrix<double>;
    using P = ublas::jacobi_precond<M>;

    M A(6, 6, 0.0);
    A(0, 0) = 4.0;
    A(0, 1) = -1.0;
    A(0, 3) = -1.0;
    A(1, 0) = -1.0;
    A(1, 1) = 4.0;
    A(1, 2) = -1.0;
    A(1, 4) = -1.0;
    A(2, 1) = -1.0;
    A(2, 2) = 4.0;
    A(2, 5) = -1.0;
    A(3, 0) = -1.0;
    A(3, 3) = 4.0;
    A(3, 4) = -1.0;
    A(4, 1) = -1.0;
    A(4, 3) = -1.0;
    A(4, 4) = 4.0;
    A(4, 5) = -1.0;
    A(5, 2) = -1.0;
    A(5, 4) = -1.0;
    A(5, 5) = 4.0;

    V sol(6, 0.0);
    sol(0) = 1.0;
    sol(1) = 2.0;
    sol(2) = 3.0;
    sol(3) = 4.0;
    sol(4) = 5.0;
    sol(5) = 6.0;

    V x(6, 0.0);
    V b = ublas::prod(A, sol);

    ublas::conjugate_gradient<M, P> solver(A);

    solver(b, x);

    EXPECT_NEAR(ublas::norm_2(x - sol), 0.0, 1e-6);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
