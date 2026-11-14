/*
 * Independent CPU reference for Knn1BucketS16: plain brute force, no buckets, int64 arithmetic.
 * Shares no code with the operator. O(n_query * n_ref): meant for small and medium cases.
 */
#ifndef KNN1_REF_HPP
#define KNN1_REF_HPP

#include <cstdint>

namespace pva_knn_ref {

/* Squared distance between query qi and ref ri; both clouds planar int16 [3][n]. */
int64_t dist2(const int16_t *q, int32_t nq, int32_t qi, const int16_t *r, int32_t nr, int32_t ri);

/* For every query: smallest squared distance to any ref, and the smallest ref index reaching it. */
void nearest(const int16_t *q, int32_t nq, const int16_t *r, int32_t nr, int64_t *outDist2, int32_t *outIdx);

} // namespace pva_knn_ref

#endif // KNN1_REF_HPP
