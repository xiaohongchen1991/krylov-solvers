/*
MIT License

Copyright (c) 2019 Xiaohong Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef BOOST_UBLAS_CONJUGATE_GRADIENT_HPP
#define BOOST_UBLAS_CONJUGATE_GRADIENT_HPP

#include "krylov_solvers_config.hpp"

#include <boost/numeric/ublas/vector.hpp>

namespace boost { namespace numeric { namespace ublas {

namespace detail {

template<class F1, class F2, class E, class V, typename Int, typename Floating>
std::tuple<Int, Floating>
conjugate_gradient_impl(const F1& A, const F2& PInv, const vector_expression<E>& b, V& x,
           Int max_iter_, Floating tol_, std::false_type) {
    typedef typename V::value_type value_type;
    typedef typename V::size_type size_type;
    typedef vector<value_type> vector_type;
    typedef std::tuple<Int, Floating> return_type;

    BOOST_UBLAS_CHECK(b().size() == x.size(), bad_size());
    BOOST_UBLAS_CHECK(max_iter_ > Int/*zero*/(), bad_argument());
    BOOST_UBLAS_CHECK(tol_ > Floating/*zero*/(), bad_argument());

    size_type max_iter = static_cast<size_type>(max_iter_);
    value_type tol = static_cast<value_type>(tol_);

    size_type n = x.size();
    // adjust max_iter
    max_iter = std::min(n, max_iter);

    value_type b_norm = norm_2(b);
    if (b_norm == value_type/*zero*/()) {
        x *= value_type/*zero*/();
        return return_type(Int/*zero*/(), Floating/*zero*/());
    }

    vector_type r = b - A(x);
    value_type error = norm_2(r) / b_norm;
    if (error < tol) {
        return return_type(Int/*zero*/(), error);
    }

    vector_type z = PInv(r);
    vector_type p(z);

    value_type r_sq_old = inner_prod(r, z);

    size_type num_iter = 1;
    for (; num_iter <= max_iter; ++num_iter) {
        vector_type Ap = A(p);
        value_type alpha = r_sq_old / inner_prod(p, Ap);
        x.plus_assign(alpha*p);
        r.minus_assign(alpha*Ap);
        z.assign(PInv(r));
        value_type r_sq_new = inner_prod(r, z);
        error = norm_2(r) / b_norm;
        if (error < tol) {
            return return_type(num_iter, error);
        }

        p = z + (r_sq_new / r_sq_old)*p;
        r_sq_old = r_sq_new;
    }

    return return_type(n, norm_2(r) / b_norm);
}

template<class M, class F, class E, class V, typename Int, typename Floating>
std::tuple<Int, Floating>
conjugate_gradient_impl(const M& A, const F& PInv, const vector_expression<E>& b, V& x,
           Int max_iter_, Floating tol_, std::true_type) {
    return conjugate_gradient_impl([&A](const auto& v){return ublas::prod(A, v);}, PInv,
                      b, x, max_iter_, tol_, std::false_type());
}

} // end namespace detail


template <class M, class F = identity_precond<M> >
class conjugate_gradient {
public:
    // param
    struct param {
        int max_iter = 10;
        int restart_iter = 10;
        double tol = 1e-5;
    };

    typedef std::tuple<int, double> return_type;

    conjugate_gradient(const M& A)
        : A_(A), PInv_(A) {}

    conjugate_gradient(const M& A, const param& p)
        : A_(A), PInv_(A), param_(p) {}

    conjugate_gradient(const M& A, const F& PInv)
        : A_(A), PInv_(PInv) {}

    conjugate_gradient(const M& A, F&& PInv)
        : A_(A), PInv_(std::move(PInv)) {}

    conjugate_gradient(const M& A, const F& PInv, const param& p)
        : A_(A), PInv_(PInv), param_(p) {}

    conjugate_gradient(const M& A, F&& PInv, const param& p)
        : A_(A), PInv_(PInv), param_(p) {}

    conjugate_gradient(const conjugate_gradient&) = delete;
    conjugate_gradient(conjugate_gradient&&) = delete;
    conjugate_gradient& operator=(const conjugate_gradient&) = delete;
    conjugate_gradient& operator=(conjugate_gradient&&) = delete;

    param get_param() const {
        return param_;
    }

    void set_param(const param& p) {
        param_ = p;
    }

    template<class E, class V>
    return_type operator()(const vector_expression<E>& b, V& x) const {
        return detail::conjugate_gradient_impl(A_, PInv_, b, x,
                                       param_.max_iter,
                                       param_.tol,
                                       detail::is_matrix_expression_t<M>());
    }

private:
    const M& A_;
    F PInv_;
    param param_;
};

} // end namespace linear
} // end namespace solver
} // end namespace math


#endif

