#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#ifdef QM_x86
#include <immintrin.h>
#endif

namespace {

struct w4a8_thread_args_4col {
    int start_j, end_j;
    const struct matmul_params *params;
};

#ifdef QM_x86
static inline float hsum_float8(__m256 v) {
    alignas(32) float buf[8];
    _mm256_storeu_ps(buf, v);
    return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
}

static inline void accumulate_one_col_4blocks(const __m256i *w_ptr, const __m256i &a0, const __m256i &a1,
                                              const __m256i &a2, const __m256i &a3, const float *s_ptr,
                                              const float *sa_ptr, __m256 &accumulator) {
    const __m256i low_mask = _mm256_set1_epi8(0xF);
    const __m256i zero_point = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);

    const __m256i raw_w = _mm256_loadu_si256(w_ptr);
    const __m256i raw_w_next = _mm256_loadu_si256(w_ptr + 1);

    const __m256i lower = _mm256_and_si256(raw_w, low_mask);
    const __m256i upper = _mm256_and_si256(_mm256_srli_epi16(raw_w, 4), low_mask);
    const __m256i lower_next = _mm256_and_si256(raw_w_next, low_mask);
    const __m256i upper_next = _mm256_and_si256(_mm256_srli_epi16(raw_w_next, 4), low_mask);

    const __m256i w0 = _mm256_sub_epi8(lower, zero_point);
    const __m256i w1 = _mm256_sub_epi8(upper, zero_point);
    const __m256i w2 = _mm256_sub_epi8(lower_next, zero_point);
    const __m256i w3 = _mm256_sub_epi8(upper_next, zero_point);

    const __m256i abs_w0 = _mm256_sign_epi8(w0, w0);
    const __m256i abs_w1 = _mm256_sign_epi8(w1, w1);
    const __m256i abs_w2 = _mm256_sign_epi8(w2, w2);
    const __m256i abs_w3 = _mm256_sign_epi8(w3, w3);

    const __m256i signed_a0 = _mm256_sign_epi8(a0, w0);
    const __m256i signed_a1 = _mm256_sign_epi8(a1, w1);
    const __m256i signed_a2 = _mm256_sign_epi8(a2, w2);
    const __m256i signed_a3 = _mm256_sign_epi8(a3, w3);

    const __m256i dot0 = _mm256_maddubs_epi16(abs_w0, signed_a0);
    const __m256i dot1 = _mm256_maddubs_epi16(abs_w1, signed_a1);
    const __m256i dot2 = _mm256_maddubs_epi16(abs_w2, signed_a2);
    const __m256i dot3 = _mm256_maddubs_epi16(abs_w3, signed_a3);

    const __m256i sum0 = _mm256_madd_epi16(ones, dot0);
    const __m256i sum1 = _mm256_madd_epi16(ones, dot1);
    const __m256i sum2 = _mm256_madd_epi16(ones, dot2);
    const __m256i sum3 = _mm256_madd_epi16(ones, dot3);

    const __m256 scale0 = _mm256_set1_ps(s_ptr[0] * sa_ptr[0]);
    const __m256 scale1 = _mm256_set1_ps(s_ptr[1] * sa_ptr[1]);
    const __m256 scale2 = _mm256_set1_ps(s_ptr[2] * sa_ptr[2]);
    const __m256 scale3 = _mm256_set1_ps(s_ptr[3] * sa_ptr[3]);

    accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum0), scale0, accumulator);
    accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum1), scale1, accumulator);
    accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum2), scale2, accumulator);
    accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum3), scale3, accumulator);
}

static inline float compute_one_output_x86(const struct matmul_params *params, int row, int col, int k,
                                           int num_block) {
    const struct matrix *A = &params->A, *B = &params->B;
    __m256 accumulator = _mm256_setzero_ps();
    const __m256i *a_ptr = (const __m256i *)&A->int8_data_ptr[row * k];
    const __m256i *w_ptr = (const __m256i *)&B->int4_data_ptr[col * k / 2];
    const float *sa_ptr = &params->A_scales[row * k / 32];
    const float *s_ptr = &params->scales[col * k / 32];

    for (int q = 0; q < num_block; q += 4) {
        const __m256i a0 = a_ptr[0];
        const __m256i a1 = a_ptr[1];
        const __m256i a2 = a_ptr[2];
        const __m256i a3 = a_ptr[3];
        accumulate_one_col_4blocks(w_ptr, a0, a1, a2, a3, s_ptr, sa_ptr, accumulator);
        a_ptr += 4;
        w_ptr += 2;
        sa_ptr += 4;
        s_ptr += 4;
    }

    return hsum_float8(accumulator);
}
#endif

static void *all_techniques_4_worker_func(void *args) {
    struct w4a8_thread_args_4col *mat_args = (struct w4a8_thread_args_4col *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int n = C->column;
    const int m = C->row;
    const int k = A->column;
    const int block_size = params->block_size;
    const int num_block = k / block_size;

    for (int row = 0; row < m; ++row) {
#ifdef QM_x86
        int col = mat_args->start_j;
        for (; col + 3 < mat_args->end_j; col += 4) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            const __m256i *a_ptr = (const __m256i *)&A->int8_data_ptr[row * k];
            const float *sa_ptr = &params->A_scales[row * k / 32];

            const __m256i *w_ptr0 = (const __m256i *)&B->int4_data_ptr[(col + 0) * k / 2];
            const __m256i *w_ptr1 = (const __m256i *)&B->int4_data_ptr[(col + 1) * k / 2];
            const __m256i *w_ptr2 = (const __m256i *)&B->int4_data_ptr[(col + 2) * k / 2];
            const __m256i *w_ptr3 = (const __m256i *)&B->int4_data_ptr[(col + 3) * k / 2];

            const float *s_ptr0 = &params->scales[(col + 0) * k / 32];
            const float *s_ptr1 = &params->scales[(col + 1) * k / 32];
            const float *s_ptr2 = &params->scales[(col + 2) * k / 32];
            const float *s_ptr3 = &params->scales[(col + 3) * k / 32];

            for (int q = 0; q < num_block; q += 4) {
                const __m256i a0 = a_ptr[0];
                const __m256i a1 = a_ptr[1];
                const __m256i a2 = a_ptr[2];
                const __m256i a3 = a_ptr[3];

                accumulate_one_col_4blocks(w_ptr0, a0, a1, a2, a3, s_ptr0, sa_ptr, acc0);
                accumulate_one_col_4blocks(w_ptr1, a0, a1, a2, a3, s_ptr1, sa_ptr, acc1);
                accumulate_one_col_4blocks(w_ptr2, a0, a1, a2, a3, s_ptr2, sa_ptr, acc2);
                accumulate_one_col_4blocks(w_ptr3, a0, a1, a2, a3, s_ptr3, sa_ptr, acc3);

                a_ptr += 4;
                sa_ptr += 4;
                w_ptr0 += 2;
                w_ptr1 += 2;
                w_ptr2 += 2;
                w_ptr3 += 2;
                s_ptr0 += 4;
                s_ptr1 += 4;
                s_ptr2 += 4;
                s_ptr3 += 4;
            }

            C->data_ptr[row * n + col + 0] = hsum_float8(acc0);
            C->data_ptr[row * n + col + 1] = hsum_float8(acc1);
            C->data_ptr[row * n + col + 2] = hsum_float8(acc2);
            C->data_ptr[row * n + col + 3] = hsum_float8(acc3);
        }

        for (; col < mat_args->end_j; ++col) {
            C->data_ptr[row * n + col] = compute_one_output_x86(params, row, col, k, num_block);
        }
#else
        for (int col = mat_args->start_j; col < mat_args->end_j; ++col) {
            float sum = 0.0f;
            const signed char *a_start = &A->int8_data_ptr[row * k];
            const unsigned char *w_start = &B->int4_data_ptr[col * k / 2];
            const float *sa_ptr = &params->A_scales[row * k / 32];
            const float *s_ptr = &params->scales[col * k / 32];

            for (int q = 0; q < num_block; ++q) {
                const float scale = sa_ptr[q] * s_ptr[q];
                for (int qi = 0; qi < block_size / 2; ++qi) {
                    const uint8_t packed = w_start[qi];
                    const int8_t w0 = (packed & 0x0F) - 8;
                    const int8_t w1 = (packed >> 4) - 8;
                    sum += (float)a_start[2 * qi] * (float)w0 * scale;
                    sum += (float)a_start[2 * qi + 1] * (float)w1 * scale;
                }
                a_start += block_size;
                w_start += block_size / 2;
            }
            C->data_ptr[row * n + col] = sum;
        }
#endif
    }

    return NULL;
}

}  // namespace

namespace matmul {

void MatmulOperator::mat_mul_all_techniques_4(struct matmul_params *params) {
    const struct matrix *A = &params->A, *C = &params->C;
    const int block_size = params->block_size;

    assert(params->block_size % 32 == 0);
    assert(A->row == C->row);
    assert(params->block_size == 32);
#ifdef QM_x86
    assert((A->column / params->block_size) % 4 == 0);
#endif

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    const int num_thread = 8;
    pthread_t thread_pool[num_thread];
    struct w4a8_thread_args_4col threads_args[num_thread];
    for (int j = 0; j < num_thread; ++j) {
        threads_args[j].start_j = (j * C->column) / num_thread;
        threads_args[j].end_j = ((j + 1) * C->column) / num_thread;
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, all_techniques_4_worker_func, &threads_args[j]);
    }

    for (int j = 0; j < num_thread; ++j) {
        pthread_join(thread_pool[j], NULL);
    }
}

}  // namespace matmul
