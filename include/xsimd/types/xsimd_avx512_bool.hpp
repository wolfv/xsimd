/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_BOOL_HPP
#define XSIMD_AVX512_BOOL_HPP

#include "xsimd_utils.hpp"

#include "xsimd_base.hpp"

namespace xsimd
{
    template <class MASK>
    class batch_bool_avx512;

    template <class MASK>
    class batch_bool_avx512
    {
    public:

        batch_bool_avx512();
        explicit batch_bool_avx512(bool b);
        batch_bool_avx512(const bool (&init)[sizeof(MASK) * 8]);

        batch_bool_avx512(const MASK& rhs);
        batch_bool_avx512& operator=(const __m512& rhs);

        bool operator[](std::size_t index) const;

        operator MASK() const;

    private:

        MASK m_value;
    };

    template <class MASK>
    inline batch_bool_avx512<MASK>::batch_bool_avx512()
    {
    }

    template <class MASK>
    inline batch_bool_avx512<MASK>::batch_bool_avx512(bool b)
        : m_value(b ? -1 : 0)
    {
    }

    namespace detail
    {
        template <class T>
        constexpr T get_init_value_impl(const bool (&/*init*/)[sizeof(T) * 8])
        {
            return T(0);
        }

        template <class T, std::size_t IX, std::size_t... I>
        constexpr T get_init_value_impl(const bool (&init)[sizeof(T) * 8])
        {
            return (init[IX] << IX) | get_init_value_impl<T, I...>(init);
        }
        
        template <class T, std::size_t... I>
        constexpr T get_init_value(const bool (&init)[sizeof(T) * 8], detail::index_sequence<I...>)
        {
            return get_init_value_impl<T, I...>(init);
        }
    }

    template <class MASK>
    inline batch_bool_avx512<MASK>::batch_bool_avx512(const bool (&init)[sizeof(MASK) * 8])
        : m_value(detail::get_init_value<MASK>(init, detail::make_index_sequence<sizeof(MASK) * 8>{}))
    {
    }

    // template <class MASK>
    // inline batch_bool_avx512<MASK>::batch_bool_avx512(const __m512d& rhs)
    //     : m_value(_mm512_cmp_pd_mask(rhs, _mm512_castsi512_pd(_mm512_set1_epi32(-1)), _CMP_EQ_OQ))
    // {
    // }

    template <class MASK>
    inline batch_bool_avx512<MASK>::batch_bool_avx512(const MASK& rhs)
        : m_value(rhs)
    {
    }

    // template <class MASK>
    // inline batch_bool_avx512<MASK>& batch_bool_avx512<MASK>::operator=(const __m512d& rhs)
    // {
    //     // m_value = rhs;
    //     m_value = _mm512_cmp_pd_mask(rhs, _mm512_castsi512_pd(_mm512_set1_epi32(-1)), _CMP_EQ_OQ);
    //     return *this;
    // }

    // template <class MASK>
    // inline batch_bool_avx512<MASK>::operator __m512d() const
    // {
    //     // TODO wrong!
    //     return (__m512d) _mm512_broadcastmb_epi64(m_value);
    //     // return m_value;
    // }

    template <class MASK>
    inline batch_bool_avx512<MASK>::operator MASK() const
    {
        return m_value;
    }

    template <class MASK>
    inline bool batch_bool_avx512<MASK>::operator[](std::size_t idx) const
    {
        return (m_value & (1 << idx)) != 0;
    }

    template <class MASK>
    inline batch_bool_avx512<MASK> operator&(const batch_bool_avx512<MASK>& lhs, const batch_bool_avx512<MASK>& rhs)
    {
        return MASK(lhs) & MASK(rhs);
    }

    template <class MASK>
    inline batch_bool_avx512<MASK> operator|(const batch_bool_avx512<MASK>& lhs, const batch_bool_avx512<MASK>& rhs)
    {
        return MASK(lhs) | MASK(rhs);
    }

    template <class MASK>
    inline batch_bool_avx512<MASK> operator^(const batch_bool_avx512<MASK>& lhs, const batch_bool_avx512<MASK>& rhs)
    {
        return MASK(lhs) ^ MASK(rhs);
    }

    template <class MASK>
    inline batch_bool_avx512<MASK> operator~(const batch_bool_avx512<MASK>& rhs)
    {
        return ~MASK(rhs);
    }

    template <class MASK>
    inline batch_bool_avx512<MASK> bitwise_andnot(const batch_bool_avx512<MASK>& lhs, const batch_bool_avx512<MASK>& rhs)
    {
        return ~(MASK(lhs) & MASK(rhs));
    }

    template <class MASK>
    inline batch_bool_avx512<MASK> operator==(const batch_bool_avx512<MASK>& lhs, const batch_bool_avx512<MASK>& rhs)
    {
        return MASK(lhs) == MASK(rhs);
    }

    template <class MASK>
    inline batch_bool_avx512<MASK> operator!=(const batch_bool_avx512<MASK>& lhs, const batch_bool_avx512<MASK>& rhs)
    {
        return MASK(lhs) != MASK(rhs);
    }

    template <class MASK>
    inline bool all(const batch_bool_avx512<MASK>& rhs)
    {
        return MASK(rhs) == MASK(-1);
    }

    template <class MASK>
    inline bool any(const batch_bool_avx512<MASK>& rhs)
    {
        return MASK(rhs) != 0;
    }
}

#endif