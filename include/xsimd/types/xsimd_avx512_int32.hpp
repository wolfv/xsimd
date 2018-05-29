/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_INT32_HPP
#define XSIMD_AVX512_INT32_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /**************************
     * batch_bool<int32_t, 16> *
     **************************/

    template <>
    struct simd_batch_traits<batch_bool<int32_t, 16>>
    {
        using value_type = bool;
        static constexpr std::size_t size = 16;
        using batch_type = batch<int32_t, 16>;
    };

    template <>
    class batch_bool<int32_t, 16> : public simd_batch_bool<batch_bool<int32_t, 16>>
    {
    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7,
                   bool b8, bool b9, bool b10, bool b11, bool b12, bool b13, bool b14, bool b15);
        batch_bool(const __m512i& rhs);
        batch_bool(const __mmask16& rhs);
        batch_bool& operator=(const __m512i& rhs);

        operator __m512i() const;

    private:

        __m512i m_value;
    };

    batch_bool<int32_t, 16> operator&(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs);
    batch_bool<int32_t, 16> operator|(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs);
    batch_bool<int32_t, 16> operator^(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs);
    batch_bool<int32_t, 16> operator~(const batch_bool<int32_t, 16>& rhs);
    batch_bool<int32_t, 16> bitwise_andnot(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs);

    batch_bool<int32_t, 16> operator==(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs);
    batch_bool<int32_t, 16> operator!=(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs);

    bool all(const batch_bool<int32_t, 16>& rhs);
    bool any(const batch_bool<int32_t, 16>& rhs);

    /*********************
     * batch<int32_t, 16> *
     *********************/

    template <>
    struct simd_batch_traits<batch<int32_t, 16>>
    {
        using value_type = int32_t;
        static constexpr std::size_t size = 16;
        using batch_bool_type = batch_bool<int32_t, 16>;
    };

    template <>
    class batch<int32_t, 16> : public simd_batch<batch<int32_t, 16>>
    {
    public:

        batch();
        explicit batch(int32_t i);
        batch(int32_t i0, int32_t i1,  int32_t i2,  int32_t i3,  int32_t i4,  int32_t i5,  int32_t i6,  int32_t i7,
              int32_t i8, int32_t i9, int32_t i10, int32_t i11, int32_t i12, int32_t i13, int32_t i14, int32_t i15);
        explicit batch(const int32_t* src);
        batch(const int32_t* src, aligned_mode);
        batch(const int32_t* src, unaligned_mode);
        batch(const __m512i& rhs);
        batch& operator=(const __m512i& rhs);

        operator __m512i() const;

        batch& load_aligned(const int32_t* src);
        batch& load_unaligned(const int32_t* src);

        batch& load_aligned(const int64_t* src);
        batch& load_unaligned(const int64_t* src);

        batch& load_aligned(const float* src);
        batch& load_unaligned(const float* src);

        batch& load_aligned(const double* src);
        batch& load_unaligned(const double* src);

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

        int32_t operator[](std::size_t index) const;

    private:

        __m512i m_value;
    };

    batch<int32_t, 16> operator-(const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> operator+(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> operator-(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> operator*(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> operator/(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);

    batch_bool<int32_t, 16> operator==(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch_bool<int32_t, 16> operator!=(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch_bool<int32_t, 16> operator<(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch_bool<int32_t, 16> operator<=(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);

    batch<int32_t, 16> operator&(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> operator|(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> operator^(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> operator~(const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> bitwise_andnot(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);

    batch<int32_t, 16> min(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> max(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);

    batch<int32_t, 16> abs(const batch<int32_t, 16>& rhs);

    batch<int32_t, 16> fma(const batch<int32_t, 16>& x, const batch<int32_t, 16>& y, const batch<int32_t, 16>& z);
    batch<int32_t, 16> fms(const batch<int32_t, 16>& x, const batch<int32_t, 16>& y, const batch<int32_t, 16>& z);
    batch<int32_t, 16> fnma(const batch<int32_t, 16>& x, const batch<int32_t, 16>& y, const batch<int32_t, 16>& z);
    batch<int32_t, 16> fnms(const batch<int32_t, 16>& x, const batch<int32_t, 16>& y, const batch<int32_t, 16>& z);

    int32_t hadd(const batch<int32_t, 16>& rhs);

    batch<int32_t, 16> select(const batch_bool<int32_t, 16>& cond, const batch<int32_t, 16>& a, const batch<int32_t, 16>& b);

    batch<int32_t, 16> operator<<(const batch<int32_t, 16>& lhs, int32_t rhs);
    batch<int32_t, 16> operator>>(const batch<int32_t, 16>& lhs, int32_t rhs);
    batch<int32_t, 16> operator<<(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> operator>>(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);

    /*****************************************
     * batch_bool<int32_t, 16> implementation *
     *****************************************/

    inline batch_bool<int32_t, 16>::batch_bool()
    {
    }

    inline batch_bool<int32_t, 16>::batch_bool(bool b)
        : m_value(_mm512_set1_epi32(-(int32_t)b))
    {
    }

    inline batch_bool<int32_t, 16>::batch_bool(const __mmask16& rhs)
        : m_value(_mm512_broadcastmw_epi32(rhs))
    {
    }


    inline batch_bool<int32_t, 16>::batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7,
                                               bool b8, bool b9, bool b10, bool b11, bool b12, bool b13, bool b14, bool b15)
        : m_value(_mm512_setr_epi32(-(int32_t)b0, -(int32_t)b1, -(int32_t)b2, -(int32_t)b3, -(int32_t)b4, -(int32_t)b5, -(int32_t)b6, -(int32_t)b7,
                                    -(int32_t)b8, -(int32_t)b9, -(int32_t)b10, -(int32_t)b11, -(int32_t)b12, -(int32_t)b13, -(int32_t)b14, -(int32_t)b15))
    {
    }

    inline batch_bool<int32_t, 16>::batch_bool(const __m512i& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<int32_t, 16>& batch_bool<int32_t, 16>::operator=(const __m512i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<int32_t, 16>::operator __m512i() const
    {
        return m_value;
    }

    inline batch_bool<int32_t, 16> operator&(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs)
    {
        return _mm512_and_si512(lhs, rhs);
    }

    inline batch_bool<int32_t, 16> operator|(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs)
    {
        return _mm512_or_si512(lhs, rhs);
    }

    inline batch_bool<int32_t, 16> operator^(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs)
    {
        return _mm512_xor_si512(lhs, rhs);
    }

    inline batch_bool<int32_t, 16> operator~(const batch_bool<int32_t, 16>& rhs)
    {
        return _mm512_xor_si512(rhs, _mm512_set1_epi32(-1));
    }

    inline batch_bool<int32_t, 16> bitwise_andnot(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs)
    {
        return _mm512_andnot_si512(lhs, rhs);
    }

    inline batch_bool<int32_t, 16> operator==(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs)
    {
        // Note this uses mask broadcasting constructor
        return _mm512_cmpeq_epi32_mask(lhs, rhs);
    }

    inline batch_bool<int32_t, 16> operator!=(const batch_bool<int32_t, 16>& lhs, const batch_bool<int32_t, 16>& rhs)
    {
        return _mm512_cmpeq_epi32_mask(lhs, rhs);
    }

    inline bool all(const batch_bool<int32_t, 16>& rhs)
    {
        _mm512_test_epi32_mask(rhs, batch_bool<int32_t, 16>(true)) != 0x0f;
    }

    inline bool any(const batch_bool<int32_t, 16>& rhs)
    {
        _mm512_test_epi64_mask(rhs, rhs) != 0;
    }

    /************************************
     * batch<int32_t, 16> implementation *
     ************************************/

    inline batch<int32_t, 16>::batch()
    {
    }

    inline batch<int32_t, 16>::batch(int32_t i)
        : m_value(_mm512_set1_epi32(i))
    {
    }

    inline batch<int32_t, 16>::batch(int32_t i0, int32_t i1,  int32_t i2,  int32_t i3,  int32_t i4,  int32_t i5,  int32_t i6,  int32_t i7,
                                     int32_t i8, int32_t i9, int32_t i10, int32_t i11, int32_t i12, int32_t i13, int32_t i14, int32_t i15)
        : m_value(_mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15))
    {
    }

    inline batch<int32_t, 16>::batch(const int32_t* src)
        : m_value(_mm512_loadu_si512((__m512i const*)src))
    {
    }

    inline batch<int32_t, 16>::batch(const int32_t* src, aligned_mode)
        : m_value(_mm512_load_si512((__m512i const*)src))
    {
    }

    inline batch<int32_t, 16>::batch(const int32_t* src, unaligned_mode)
        : m_value(_mm512_loadu_si512((__m512i const*)src))
    {
    }

    inline batch<int32_t, 16>::batch(const __m512i& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::operator=(const __m512i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int32_t, 16>::operator __m512i() const
    {
        return m_value;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const int32_t* src)
    {
        m_value = _mm512_load_si512((__m512i const*)src);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const int32_t* src)
    {
        m_value = _mm512_loadu_si512((__m512i const*)src);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const int64_t* src)
    {
        alignas(64) int32_t tmp[16];
        tmp[0] = static_cast<int32_t>(src[0]);
        tmp[1] = static_cast<int32_t>(src[1]);
        tmp[2] = static_cast<int32_t>(src[2]);
        tmp[3] = static_cast<int32_t>(src[3]);
        tmp[4] = static_cast<int32_t>(src[4]);
        tmp[5] = static_cast<int32_t>(src[5]);
        tmp[6] = static_cast<int32_t>(src[6]);
        tmp[7] = static_cast<int32_t>(src[7]);
        tmp[8] = static_cast<int32_t>(src[8]);
        tmp[9] = static_cast<int32_t>(src[9]);
        tmp[10] = static_cast<int32_t>(src[10]);
        tmp[11] = static_cast<int32_t>(src[11]);
        tmp[12] = static_cast<int32_t>(src[12]);
        tmp[13] = static_cast<int32_t>(src[13]);
        tmp[14] = static_cast<int32_t>(src[14]);
        tmp[15] = static_cast<int32_t>(src[15]);
        m_value = _mm512_load_si512((__m512i const*)tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const float* src)
    {
        m_value = _mm512_cvtps_epi32(_mm512_load_ps(src));
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const float* src)
    {
        m_value = _mm512_cvtps_epi32(_mm512_loadu_ps(src));
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const double* src)
    {
        __m256i tmp1 = _mm512_cvtpd_epi32(_mm512_load_pd(src));
        __m256i tmp2 = _mm512_cvtpd_epi32(_mm512_load_pd(src + 8));
        m_value = _mm512_castsi256_si512(tmp1);
        m_value = _mm512_inserti32x8(m_value, tmp2, 1);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const double* src)
    {
        __m256i tmp1 = _mm512_cvtpd_epi32(_mm512_loadu_pd(src));
        __m256i tmp2 = _mm512_cvtpd_epi32(_mm512_loadu_pd(src + 8));
        m_value = _mm512_castsi256_si512(tmp1);
        m_value = _mm512_inserti32x8(m_value, tmp2, 1);
        return *this;
    }

    inline void batch<int32_t, 16>::store_aligned(int32_t* dst) const
    {
        _mm512_store_si512((__m512i*)dst, m_value);
    }

    inline void batch<int32_t, 16>::store_unaligned(int32_t* dst) const
    {
        _mm512_storeu_si512((__m512i*)dst, m_value);
    }

    inline void batch<int32_t, 16>::store_aligned(int64_t* dst) const
    {
        alignas(64) int32_t tmp[16];
        store_aligned(tmp);
        dst[0] = int64_t(tmp[0]);
        dst[1] = int64_t(tmp[1]);
        dst[2] = int64_t(tmp[2]);
        dst[3] = int64_t(tmp[3]);
        dst[4] = int64_t(tmp[4]);
        dst[5] = int64_t(tmp[5]);
        dst[6] = int64_t(tmp[6]);
        dst[7] = int64_t(tmp[7]);
        dst[8] = int64_t(tmp[8]);
        dst[9] = int64_t(tmp[9]);
        dst[10] = int64_t(tmp[10]);
        dst[11] = int64_t(tmp[11]);
        dst[12] = int64_t(tmp[12]);
        dst[13] = int64_t(tmp[13]);
        dst[14] = int64_t(tmp[14]);
        dst[15] = int64_t(tmp[15]);
    }

    inline void batch<int32_t, 16>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int32_t, 16>::store_aligned(float* dst) const
    {
        _mm512_store_ps(dst, _mm512_cvtepi32_ps(m_value));
    }

    inline void batch<int32_t, 16>::store_unaligned(float* dst) const
    {
        _mm512_storeu_ps(dst, _mm512_cvtepi32_ps(m_value));
    }

    inline void batch<int32_t, 16>::store_aligned(double* dst) const
    {
        __m256i tmp1 = _mm512_extracti32x8_epi32(m_value, 0);
        __m256i tmp2 = _mm512_extracti32x8_epi32(m_value, 1);
        _mm512_store_pd(dst, _mm512_cvtepi32_pd(tmp1));
        _mm512_store_pd(dst + 8 , _mm512_cvtepi32_pd(tmp2));
    }

    inline void batch<int32_t, 16>::store_unaligned(double* dst) const
    {
        __m256i tmp1 = _mm512_extracti32x8_epi32(m_value, 0);
        __m256i tmp2 = _mm512_extracti32x8_epi32(m_value, 1);
        _mm512_store_pd(dst, _mm512_cvtepi32_pd(tmp1));
        _mm512_store_pd(dst + 8 , _mm512_cvtepi32_pd(tmp2));
    }

    inline int32_t batch<int32_t, 16>::operator[](std::size_t index) const
    {
        alignas(64) int32_t x[16];
        store_aligned(x);
        return x[index & 15];
    }

    inline batch<int32_t, 16> operator-(const batch<int32_t, 16>& rhs)
    {
        return _mm512_sub_epi32(_mm512_setzero_si512(), rhs);
    }

    inline batch<int32_t, 16> operator+(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_add_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> operator-(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_sub_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> operator*(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_mullo_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> operator/(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_cvttps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(lhs), _mm512_cvtepi32_ps(rhs)));
    }

    inline batch_bool<int32_t, 16> operator==(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_cmp_epu32_mask(lhs, rhs, _MM_CMPINT_EQ);
    }

    inline batch_bool<int32_t, 16> operator!=(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_cmp_epu32_mask(lhs, rhs, _MM_CMPINT_NE);
    }

    inline batch_bool<int32_t, 16> operator<(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_cmp_epu32_mask(lhs, rhs, _MM_CMPINT_GT);
    }

    inline batch_bool<int32_t, 16> operator<=(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_cmp_epu32_mask(lhs, rhs, _MM_CMPINT_GE);
    }

    inline batch<int32_t, 16> operator&(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_and_si512(lhs, rhs);
    }

    inline batch<int32_t, 16> operator|(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_or_si512(lhs, rhs);
    }

    inline batch<int32_t, 16> operator^(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_xor_si512(lhs, rhs);
    }

    inline batch<int32_t, 16> operator~(const batch<int32_t, 16>& rhs)
    {
        return _mm512_xor_si512(rhs, _mm512_set1_epi32(-1));
    }

    inline batch<int32_t, 16> bitwise_andnot(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_andnot_si512(lhs, rhs);
    }

    inline batch<int32_t, 16> min(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_min_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> max(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_max_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> abs(const batch<int32_t, 16>& rhs)
    {
        return _mm512_abs_epi32(rhs);
    }

    inline batch<int32_t, 16> fma(const batch<int32_t, 16>& x, const batch<int32_t, 16>& y, const batch<int32_t, 16>& z)
    {
        // Note: support for _mm512_fmadd_epi32 in KNC ?
        return x * y + z;
    }

    inline batch<int32_t, 16> fms(const batch<int32_t, 16>& x, const batch<int32_t, 16>& y, const batch<int32_t, 16>& z)
    {
        return x * y - z;
    }

    inline batch<int32_t, 16> fnma(const batch<int32_t, 16>& x, const batch<int32_t, 16>& y, const batch<int32_t, 16>& z)
    {
        return -x * y + z;
    }

    inline batch<int32_t, 16> fnms(const batch<int32_t, 16>& x, const batch<int32_t, 16>& y, const batch<int32_t, 16>& z)
    {
        return -x * y - z;
    }

    inline int32_t hadd(const batch<int32_t, 16>& rhs)
    {
        return _mm512_reduce_add_epi32(rhs);
    }

    inline batch<int32_t, 16> select(const batch_bool<int32_t, 16>& cond, const batch<int32_t, 16>& a, const batch<int32_t, 16>& b)
    {
        // BIG QUESTIONMARK
        // return _mm512_mask_blend_epi32(cond, a, b);
    }

    inline batch<int32_t, 16> operator<<(const batch<int32_t, 16>& lhs, int32_t rhs)
    {
        return _mm512_slli_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> operator>>(const batch<int32_t, 16>& lhs, int32_t rhs)
    {
        return _mm512_srli_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> operator<<(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_sllv_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> operator>>(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_srlv_epi32(lhs, rhs);
    }
}

#endif
