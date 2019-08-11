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

#ifndef BOOST_UBLAS_PRECONDITIONER_HPP
#define BOOST_UBLAS_PRECONDITIONER_HPP

#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace boost { namespace numeric { namespace ublas {

template <class M>
class identity_precond {
public:
    identity_precond() {}
    identity_precond(const M&) {}

    template<class E>
    const vector_expression<E>& operator()(const vector_expression<E>& ve) const {
        return ve;
    }
};

template <class M>
class jacobi_precond {
public:
    typedef typename M::value_type value_type;
    typedef typename M::size_type size_type;
    jacobi_precond(const M& A)
        : diag_(A.size1()) {
        size_type size = A.size1();
        for (size_type i = 0; i < size; ++i) {
            diag_(i) = A(i, i);
        }
    }

    jacobi_precond(const jacobi_precond&) = default;
    jacobi_precond(jacobi_precond&&) = default;
    jacobi_precond& operator=(const jacobi_precond&) = default;
    jacobi_precond& operator=(jacobi_precond&&) = default;

    template<class E>
    decltype(auto) operator()(const vector_expression<E>& ve) const {
        return element_prod(ve, diag_);
    }

private:
    vector<value_type> diag_;
};

} // end namespace linear
} // end namespace solver
} // end namespace math


#endif
