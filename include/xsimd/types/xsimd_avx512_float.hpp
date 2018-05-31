/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_FLOAT_HPP
#define XSIMD_AVX512_FLOAT_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /*************************
     * batch_bool<float, 16> *
     *************************/

    template <>
    struct simd_batch_traits<batch_bool<float, 16>>
    {
        using value_type = bool;
        static constexpr std::size_t size = 16;
        using batch_type = batch<float, 16>;
    };

    template <>
    class batch_bool<float, 16> : 
        public batch_bool_avx512<__mmask16, batch_bool<float, 16>>,
        public simd_batch_bool<batch_bool<float, 16>>
    {
    public:
        using base_class = batch_bool_avx512<__mmask16, batch_bool<float, 16>>;
        using base_class::base_class;

        batch_bool(bool b0, bool b1,  bool b2,  bool b3,  bool b4,  bool b5,  bool b6,  bool b7,
                   bool b8, bool b9, bool b10, bool b11, bool b12, bool b13, bool b14, bool b15)
            : base_class({{b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}})
        {
        }
    };

    GENERATE_AVX512_BOOL_OPS(float, 16);

    /*********************
     * batch<float, 16> *
     *********************/

    template <>
    struct simd_batch_traits<batch<float, 16>>
    {
        using value_type = float;
        using batch_bool_type = batch_bool<float, 16>;
        static constexpr std::size_t size = 16;
    };

    template <>
    class batch<float, 16> : public simd_batch<batch<float, 16>>
    {
    public:

        batch();
        explicit batch(float i);
        batch(float i0, float i1,  float i2,  float i3,  float i4,  float i5,  float i6,  float i7,
              float i8, float i9, float i10, float i11, float i12, float i13, float i14, float i15);
        explicit batch(const float* src);
        batch(const float* src, aligned_mode);
        batch(const float* src, unaligned_mode);
        batch(const __m512& rhs);
        batch& operator=(const __m512& rhs);

        operator __m512() const;

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

        float operator[](std::size_t index) const;

    private:

        __m512 m_value;
    };

    batch<float, 16> operator-(const batch<float, 16>& rhs);
    batch<float, 16> operator+(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator-(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator*(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator/(const batch<float, 16>& lhs, const batch<float, 16>& rhs);

    batch_bool<float, 16> operator==(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch_bool<float, 16> operator!=(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch_bool<float, 16> operator<(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch_bool<float, 16> operator<=(const batch<float, 16>& lhs, const batch<float, 16>& rhs);

    batch<float, 16> operator&(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator|(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator^(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> operator~(const batch<float, 16>& rhs);
    batch<float, 16> bitwise_andnot(const batch<float, 16>& lhs, const batch<float, 16>& rhs);

    batch<float, 16> min(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> max(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> fmin(const batch<float, 16>& lhs, const batch<float, 16>& rhs);
    batch<float, 16> fmax(const batch<float, 16>& lhs, const batch<float, 16>& rhs);

    batch<float, 16> abs(const batch<float, 16>& rhs);
    batch<float, 16> fabs(const batch<float, 16>& rhs);
    batch<float, 16> sqrt(const batch<float, 16>& rhs);

    batch<float, 16> fma(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z);
    batch<float, 16> fms(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z);
    batch<float, 16> fnma(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z);
    batch<float, 16> fnms(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z);

    float hadd(const batch<float, 16>& rhs);
    batch<double, 8> haddp(const batch<double, 8>* row);

    batch<float, 16> select(const batch_bool<float, 16>& cond, const batch<float, 16>& a, const batch<float, 16>& b);

    batch_bool<double, 8> isnan(const batch<double, 8>& x);

    /************************************
     * batch<float, 16> implementation *
     ************************************/

    inline batch<float, 16>::batch()
    {
    }

    inline batch<float, 16>::batch(float i)
        : m_value(_mm512_set1_ps(i))
    {
    }

    inline batch<float, 16>::batch(float i0, float i1,  float i2,  float i3,  float i4,  float i5,  float i6,  float i7,
                                   float i8, float i9, float i10, float i11, float i12, float i13, float i14, float i15)
        : m_value(_mm512_setr_ps(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15))
    {
    }

    inline batch<float, 16>::batch(const float* src)
        : m_value(_mm512_loadu_ps(src))
    {
    }

    inline batch<float, 16>::batch(const float* src, aligned_mode)
        : m_value(_mm512_load_ps(src))
    {
    }

    inline batch<float, 16>::batch(const float* src, unaligned_mode)
        : m_value(_mm512_load_ps(src))
    {
    }

    inline batch<float, 16>::batch(const __m512& rhs)
        : m_value(rhs)
    {
    }

    inline batch<float, 16>& batch<float, 16>::operator=(const __m512& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<float, 16>::operator __m512() const
    {
        return m_value;
    }

    inline batch<float, 16>& batch<float, 16>::load_aligned(const float* src)
    {
        m_value = _mm512_load_ps(src);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_unaligned(const float* src)
    {
        m_value = _mm512_loadu_ps(src);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_aligned(const int64_t* src)
    {
        alignas(64) float tmp[16];
        tmp[0] = static_cast<float>(src[0]);
        tmp[1] = static_cast<float>(src[1]);
        tmp[2] = static_cast<float>(src[2]);
        tmp[3] = static_cast<float>(src[3]);
        tmp[4] = static_cast<float>(src[4]);
        tmp[5] = static_cast<float>(src[5]);
        tmp[6] = static_cast<float>(src[6]);
        tmp[7] = static_cast<float>(src[7]);
        tmp[8] = static_cast<float>(src[8]);
        tmp[9] = static_cast<float>(src[9]);
        tmp[10] = static_cast<float>(src[10]);
        tmp[11] = static_cast<float>(src[11]);
        tmp[12] = static_cast<float>(src[12]);
        tmp[13] = static_cast<float>(src[13]);
        tmp[14] = static_cast<float>(src[14]);
        tmp[15] = static_cast<float>(src[15]);
        m_value = _mm512_load_ps(tmp);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline batch<float, 16>& batch<float, 16>::load_aligned(const int32_t* src)
    {
        // TODO select correct rounding direction?
        m_value = _mm512_cvt_roundepi32_ps(_mm512_load_si512(src), _MM_FROUND_CUR_DIRECTION);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_unaligned(const int32_t* src)
    {
        m_value = _mm512_cvt_roundepi32_ps(_mm512_loadu_si512(src), _MM_FROUND_CUR_DIRECTION);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_aligned(const double* src)
    {
        __m256 tmp1 = _mm512_cvtpd_ps(_mm512_load_pd(src));
        __m256 tmp2 = _mm512_cvtpd_ps(_mm512_load_pd(src + 8));
        m_value = _mm512_castps256_ps512(tmp1);
        m_value = _mm512_insertf32x8(m_value, tmp2, 1);
        return *this;
    }

    inline batch<float, 16>& batch<float, 16>::load_unaligned(const double* src)
    {
        __m256 tmp1 = _mm512_cvtpd_ps(_mm512_loadu_pd(src));
        __m256 tmp2 = _mm512_cvtpd_ps(_mm512_loadu_pd(src + 8));
        m_value = _mm512_castps256_ps512(tmp1);
        m_value = _mm512_insertf32x8(m_value, tmp2, 1);
        return *this;
    }

    inline void batch<float, 16>::store_aligned(int32_t* dst) const
    {
        _mm512_store_si512((__m512i *)dst, _mm512_cvtps_epi32(m_value));
    }

    inline void batch<float, 16>::store_unaligned(int32_t* dst) const
    {
        _mm512_storeu_si512((__m512i *)dst, _mm512_cvtps_epi32(m_value));
    }

    inline void batch<float, 16>::store_aligned(int64_t* dst) const
    {
        alignas(64) float tmp[16];
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

    inline void batch<float, 16>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<float, 16>::store_aligned(float* dst) const
    {
        _mm512_store_ps(dst, m_value);
    }

    inline void batch<float, 16>::store_unaligned(float* dst) const
    {
        _mm512_storeu_ps(dst, m_value);
    }

    inline void batch<float, 16>::store_aligned(double* dst) const
    {
        _mm512_store_pd(dst, _mm512_cvtps_pd(_mm512_extractf32x8_ps(m_value, 0)));
        _mm512_store_pd(dst + 8, _mm512_cvtps_pd(_mm512_extractf32x8_ps(m_value, 1)));
    }

    inline void batch<float, 16>::store_unaligned(double* dst) const
    {
        _mm512_storeu_pd(dst, _mm512_cvtps_pd(_mm512_extractf32x8_ps(m_value, 0)));
        _mm512_storeu_pd(dst + 8, _mm512_cvtps_pd(_mm512_extractf32x8_ps(m_value, 1)));
    }

    inline float batch<float, 16>::operator[](std::size_t index) const
    {
        alignas(64) float x[16];
        store_aligned(x);
        return x[index & 15];
    }

    inline batch<float, 16> operator-(const batch<float, 16>& rhs)
    {
        return _mm512_sub_ps(_mm512_setzero_ps(), rhs);
    }

    inline batch<float, 16> operator+(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_add_ps(lhs, rhs);
    }

    inline batch<float, 16> operator-(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_sub_ps(lhs, rhs);
    }

    inline batch<float, 16> operator*(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_mul_ps(lhs, rhs);
    }

    inline batch<float, 16> operator/(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_div_ps(lhs, rhs);
    }

    inline batch_bool<float, 16> operator==(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_EQ_OQ);
    }

    inline batch_bool<float, 16> operator!=(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_NEQ_OQ);
    }

    inline batch_bool<float, 16> operator<(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_LT_OQ);
    }

    inline batch_bool<float, 16> operator<=(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_LE_OQ);
    }

    inline batch<float, 16> operator&(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_and_ps(lhs, rhs);
    }

    inline batch<float, 16> operator|(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_or_ps(lhs, rhs);
    }

    inline batch<float, 16> operator^(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_xor_ps(lhs, rhs);
    }

    inline batch<float, 16> operator~(const batch<float, 16>& rhs)
    {
        return _mm512_xor_ps(rhs, _mm512_castsi512_ps(_mm512_set1_epi32(-1)));
    }

    inline batch<float, 16> bitwise_andnot(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_andnot_ps(lhs, rhs);
    }

    inline batch<float, 16> min(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_min_ps(lhs, rhs);
    }

    inline batch<float, 16> max(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return _mm512_max_ps(lhs, rhs);
    }

    inline batch<float, 16> fmin(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return min(lhs, rhs);
    }

    inline batch<float, 16> fmax(const batch<float, 16>& lhs, const batch<float, 16>& rhs)
    {
        return max(lhs, rhs);
    }

    inline batch<float, 16> abs(const batch<float, 16>& rhs)
    {
        // return _mm512_abs_ps(rhs);
        return static_cast<__m512>(_mm512_and_epi32(static_cast<__m512i>(static_cast<__m512>(rhs)),
                                                    _mm512_set1_epi32(0x7fffffff)));
    }

    inline batch<float, 16> sqrt(const batch<float, 16>& rhs)
    {
        return _mm512_sqrt_ps(rhs);
    }

    inline batch<float, 16> fabs(const batch<float, 16>& rhs)
    {
        return abs(rhs);
    }

    inline batch<float, 16> fma(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z)
    {
        return _mm512_fmadd_ps(x, y, z);
    }

    inline batch<float, 16> fms(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z)
    {
        return _mm512_fmsub_ps(x, y, z);
    }

    inline batch<float, 16> fnma(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z)
    {
        return _mm512_fnmadd_ps(x, y, z);
    }

    inline batch<float, 16> fnms(const batch<float, 16>& x, const batch<float, 16>& y, const batch<float, 16>& z)
    {
        return _mm512_fnmsub_ps(x, y, z);
    }

    inline float hadd(const batch<float, 16>& rhs)
    {
        // return _mm512_reduce_add_ps(rhs);
        __m256 tmp1 = _mm512_extractf32x8_ps(rhs, 1);
        __m256 tmp2 = _mm512_extractf32x8_ps(rhs, 0);
        __m256 res1 = tmp1 + tmp2;
        return hadd(batch<float, 8>(res1));
    }

    inline batch<float, 16> select(const batch_bool<float, 16>& cond, const batch<float, 16>& a, const batch<float, 16>& b)
    {
        return _mm512_mask_blend_ps(cond, b, a);
    }

    inline batch_bool<float, 16> isnan(const batch<float, 16>& x)
    {
        return _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
    }
}

#endif