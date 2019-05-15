// Preconditioners for linear solver
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
