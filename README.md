# Release versions

version 1.0.0:

The libaray includes two solvers, gmres and conjudate gradient methods, and two preconditioners, identity and Jocobi preconditioners.

# Quick start

This is a simple C++ library of some commonly used Krylov subspace methods for solving linear systems. The algorithms and APIs are designed to support boost.ublas containers.

## Basics

    #include <gmres.hpp>
    namespace ublas=boost::numeric::ublas;

Create and initialize a ublas::matrix instance.

    using M = ublas::matrix<double>;
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

Create and initialize a ublas::gmres instance.

    ublas::gmres<M> solver(A);

The function call operator is overloaded to solve the linear system Ax = b.

    using V = ublas::vector<double>;
    V b(3);
    b(0) = 14.0;
    b(1) = 32.0;
    b(2) = 50.0;
    V x(3, 0.0);
    auto result = solver(b, x);

To get the number of iterations and error, do

    int num_iter = std::get<0>(result);
    double error = std::get<1>(result);

## Sparse matrix

The krylov-solvers libaray also supports boost.ublas sparse matrix containers.

    #include <conjugate_gradient.hpp>
    #include <boost/numeric/ublas/matrix_sparse.hpp>
    namespace ublas=boost::numeric::ublas;

Create a sparse matrix.

    using M = ublas::compressed_matrix<double>;
    M A(3, 3);
    A(0, 0) = -2.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = -2.0;
    A(1, 2) = 1.0;
    A(2, 1) = 1.0;
    A(2, 2) = -2.0;

Create a ublas::conjugate_gradient instance.

    ublas::conjugate_gradient<M> solver(A);

Solve linear system Ax = b.

    using V = ublas::vector<double>;
    V b(3);
    b(0) = 0.0;
    b(1) = 0.0;
    b(2) = -4.0;
    V x(3, 0.0);
    solver(b, x);

## Config solver parameters

Each solver class defines a nested struct "param". One way to config solver is to pass a param struct into its constructor. e.g.

    typename ublas::gmres<M>::param p;
    p.tol = 1e-8;
    ublas::gmres<M> solver(A, p);

The solver parameters can also be set by doing:

    solver.set_param(p);

## Preconditioner

Two simple preconditioners are provided in "preconditioner.hpp" file, namely, identity preconditioner and Jacobi preconditioner. By default, identity preconditioner is used. To explicitly specify the preconditioner, do:

    using M = ublas::compressed_matrix<double>;
    using P = ublas::jacobi_precond<M>;
    ublas::conjugate_gradient<M, P> solver(A);

## Customization

The Krylov subspace methods are iterative methods and do not require forming the matrix A explicitly. So the user can also initialize the solver by a functor which computes Ax. In a similar way, one can also specify a customized preconditioner. e.g.

    // functor to compute Ax
    auto Op = [&A](const auto& v){return ublas::prod(A, v);};
    // customized preconditioner
    auto PInv = [](const auto& v)->decltype(auto){return v;};
    ublas::conjugate_gradient<decltype(Op), decltype(PInv)> solver(Op, PInv);

## Linking

This is a header only library.

## Requirements

The libarary requires a C++ compiler that supports C++14 and boost.ublas. To build the test, add boost include directory into system path.

## TODO list

* Add bicgstab solver
* Add ilu0 preconditioner
