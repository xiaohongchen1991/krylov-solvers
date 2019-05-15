// GMRES iterative solver
// Copyright (C) 2019  Xiaohong Chen

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// E-mail: xiaohong_chen1991@hotmail.com

#ifndef BOOST_UBLAS_GMRES_HPP
#define BOOST_UBLAS_GMRES_HPP

#include "preconditioner.hpp"
#include "krylov_solvers_config.hpp"

#include <tuple>
#include <cmath>
#include <type_traits>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/operation.hpp>

// test
#include <iostream>
#include <boost/numeric/ublas/io.hpp>

namespace boost { namespace numeric { namespace ublas {

namespace detail {
// Calculate the Given rotation matrix
template<typename T>
BOOST_UBLAS_INLINE std::tuple<T, T> givens_rotation(T x1, T x2) {
    T d = sqrt(x1*x1 + x2*x2);
    if (d == T/*zero*/()) divide_by_zero().raise();
    return std::make_tuple(x1/d, -x2/d);
}

// apply Givens rotations on the jth column of the upper Hessenberg matrix.
template<class V1, class V2>
void apply_givens_rotation(V1& h, V2& cs, V2& sn,
                           typename vector_traits<V2>::size_type j) {
    typedef typename V2::size_type size_type;
    typedef typename V2::value_type value_type;

    BOOST_UBLAS_CHECK (cs.size() == sn.size(), bad_size ());
    BOOST_UBLAS_CHECK (cs.size() > j, bad_size ());

    // apply Givens rotation over the first j - 1 entries on jth column of the upper Hessenberg matrix.
    for (size_type i = 0; i < j; ++i) {
        value_type tmp = cs(i)*h(i) - sn(i)*h(i+1);
        h(i+1) = sn(i)*h(i) + cs(i)*h(i+1);
        h(i) = tmp;
    }
    // update Givens rotation matrix.
    if (j < h.size() - 1) {
        BOOST_UBLAS_CHECK (h.size() > j + 1, bad_size ());
        std::tie(cs(j), sn(j)) = givens_rotation(h(j), h(j+1));
        // eliminate h(j+1).
        h(j) = cs(j)*h(j) - sn(j)*h(j+1);
        h(j+1) = value_type/*zero*/();
    }
    else {
        cs(j) = value_type(1.0);
        sn(j) = value_type/*zero*/();
    }
}

template<class F1, class F2, class M, class V>
void arnoldi(const F1& A, const F2& PInv, M& Q, V& h, typename matrix_traits<M>::size_type j) {
    typedef typename M::size_type size_type;
    typedef typename M::value_type value_type;
    typedef M matrix_type;
    typedef vector<value_type> vector_type;

    BOOST_UBLAS_CHECK(Q.size2() > j + 1, bad_size());

    matrix_column<matrix_type> q(Q, j+1);
    q.assign(PInv(A(column(Q, j))));
    size_type size1 = Q.size1();
    vector_range<V> hr(h, range(0, j+1));
    // CGS2
    axpy_prod(q, project(Q, range(0, size1), range(0, j+1)), hr, true);
    axpy_prod(project(Q, range(0, size1), range(0, j+1)), -hr, q, false);
    vector_type tmp(j+1, 0.0);
    axpy_prod(q, project(Q, range(0, size1), range(0, j+1)), tmp, false);
    axpy_prod(project(Q, range(0, size1), range(0, j+1)), -tmp, q, false);
    hr.plus_assign(tmp);

    if (j < size1 - 1) {
        BOOST_UBLAS_CHECK(h.size() > j + 1, bad_size());
        h(j+1) = norm_2(q);
        q /= h(j+1);
    }
    else {
        q *= value_type/*zero*/();
    }
}

template <typename T>
class is_matrix_expression
{
private:
    template<typename E>
    static constexpr std::true_type  test(const matrix_expression<E> *);
    static constexpr std::false_type test(...);
public:
    static constexpr bool value = decltype(test(std::declval<T*>()))::value;
    typedef decltype(test(std::declval<T*>())) type;
};

template <typename T>
constexpr bool is_matrix_expression_v = is_matrix_expression<T>::value;

template <typename T>
using is_matrix_expression_t = typename is_matrix_expression<T>::type;

template<class F1, class F2, class E, class V, typename Int, typename Floating>
std::tuple<Int, Floating>
gmres_impl(const F1& A, const F2& PInv, const vector_expression<E>& b, V& x,
           Int max_iter_, Int restart_iter_, Floating tol_, std::false_type) {
    typedef typename V::value_type value_type;
    typedef typename V::size_type size_type;
    typedef matrix<value_type, column_major> matrix_type;
    typedef banded_matrix<value_type, column_major> banded_matrix_type;
    typedef vector<value_type> vector_type;
    typedef unit_vector<value_type> unit_vector_type;
    typedef std::tuple<Int, Floating> return_type;

    BOOST_UBLAS_CHECK(b().size() == x.size(), bad_size());
    BOOST_UBLAS_CHECK(restart_iter_ > Int/*zero*/(), bad_argument());
    BOOST_UBLAS_CHECK(max_iter_ > Int/*zero*/(), bad_argument());
    BOOST_UBLAS_CHECK(tol_ > Floating/*zero*/(), bad_argument());

    size_type n = x.size();
    // adjust max_iter and restart_iter
    size_type max_iter = std::min(n, static_cast<size_type>(max_iter_));
    size_type restart_iter = std::min(n, static_cast<size_type>(restart_iter_));
    value_type tol = static_cast<value_type>(tol_);
    if (restart_iter > max_iter) restart_iter = max_iter;

    vector_type r = PInv(b - A(x));
    value_type r_norm = norm_2(r);
    value_type b_norm = norm_2(PInv(b));
    if (b_norm == value_type/*zero*/()) {
        x *= value_type/*zero*/();
        return return_type(Int/*zero*/(), Floating/*zero*/());
    }
    value_type error = norm_2(r) / b_norm;
    if (error < tol) return return_type(Int/*zero*/(), error);

    vector_type sn(restart_iter, value_type/*zero*/());
    vector_type cs(restart_iter, value_type/*zero*/());

    matrix_type Q(n, restart_iter + 1, value_type/*zero*/());
    banded_matrix_type H(n, restart_iter, 1, restart_iter);

    size_type num_iter = 1;
    while (num_iter <= max_iter) {
        column(Q, 0).assign(r / r_norm);
        vector_type beta = r_norm * unit_vector_type(restart_iter + 1, 0);

        size_type j = size_type/*zero*/();
        for (; j < restart_iter && num_iter <= max_iter; ++j, ++num_iter) {
            // run arnoldi
            matrix_column<banded_matrix_type> H_j(H, j);
            detail::arnoldi(A, PInv, Q, H_j, j);

            // eliminate the last element in H jth column and update the rotation matrix
            detail::apply_givens_rotation(H_j, cs, sn, j);

            // update the residual vector
            beta(j+1) = sn(j)*beta(j);
            beta(j) = cs(j)*beta(j);
            error  = std::abs(beta(j+1)) / b_norm;

            if (error <= tol) {
                // update x
                auto y = solve(project(H, range(0, j+1), range(0, j+1)),
                               project(beta, range(0, j+1)), upper_tag());
                axpy_prod(project(Q, range(0, n), range(0, j+1)), y, x, false);
                return return_type(num_iter, error);
            }
        }
        // update x
        auto y = solve(project(H, range(0, j), range(0, j)),
                       project(beta, range(0, j)), upper_tag());
        axpy_prod(project(Q, range(0, n), range(0, j)), y, x, false);

        r = PInv(b - A(x));
        r_norm = norm_2(r);
        error = norm_2(r) / b_norm;
        if (error < tol) return return_type(num_iter, error);
    }
    return return_type(max_iter, error);
}

template<class M, class F, class E, class V, typename Int, typename Floating>
std::tuple<Int, Floating>
gmres_impl(const M& A, const F& PInv, const vector_expression<E>& b, V& x,
           Int max_iter, Int restart_iter, Floating tol, std::true_type) {
    return gmres_impl([&A](const auto& v){return ublas::prod(A, v);}, PInv,
                      b, x, max_iter, restart_iter, tol, std::false_type());
}

} // end namespace detail


template <class M, class F = identity_precond<M> >
class gmres {
public:
    // param
    struct param {
        int max_iter = 10;
        int restart_iter = 10;
        double tol = 1e-6;
    };

    typedef std::tuple<int, double> return_type;

    gmres(const M& A)
        : A_(A), PInv_(A) {}

    gmres(const M& A, const param& p)
        : A_(A), PInv_(A), param_(p) {}

    gmres(const M& A, const F& PInv)
        : A_(A), PInv_(PInv) {}

    gmres(const M& A, F&& PInv)
        : A_(A), PInv_(std::move(PInv)) {}

    gmres(const M& A, const F& PInv, const param& p)
        : A_(A), PInv_(PInv), param_(p) {}

    gmres(const M& A, F&& PInv, const param& p)
        : A_(A), PInv_(PInv), param_(p) {}

    gmres(const gmres&) = delete;
    gmres(gmres&&) = delete;
    gmres& operator=(const gmres&) = delete;
    gmres& operator=(gmres&&) = delete;

    param get_param() const {
        return param_;
    }

    void set_param(const param& p) {
        param_ = p;
    }

    template<class E, class V>
    return_type operator()(const vector_expression<E>& b, V& x) const {
        return detail::gmres_impl(A_, PInv_, b, x,
                                  param_.max_iter,
                                  param_.restart_iter,
                                  param_.tol,
                                  detail::is_matrix_expression_t<M>());
    }

private:
    const M& A_;
    F PInv_;
    param param_;
};

} // end namespace ublas
} // end namespace numeric
} // end namespace boost

#endif
