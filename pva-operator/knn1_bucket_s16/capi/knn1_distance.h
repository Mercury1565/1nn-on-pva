/* knn1_distance: plain C interface to the exact 1-NN PVA operator (no cuPVA headers needed).
 *     knn1_ctx *ctx = knn1_create();                          // once
 *     knn1_nearest(ctx, q, nq, r, nr, dist2, idx, &sum_d2);   // cloud distance = sqrt(sum_d2) cm
 *     knn1_destroy(ctx);
 * One call at a time per context. Functions returning int return 0 on success. */
#ifndef KNN1_DISTANCE_H
#define KNN1_DISTANCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct knn1_ctx knn1_ctx;

/* Create a context: allocates the PVA buffers and loads the PVA program. NULL on failure. */
knn1_ctx *knn1_create(void);

/* Release a context (NULL is allowed). */
void knn1_destroy(knn1_ctx *ctx);

/*
 * Exact 1-NN of every query point in pc_ref. Blocks until the PVA has finished.
 *   pc_queries  int16[3 * n_query], planar, cm
 *   pc_ref      int16[3 * n_ref],   planar, cm
 *   dist2       int32[n_query] out: squared distance to the nearest ref point (cm^2)
 *   idx         int32[n_query] out: index of that ref point in pc_ref (may be NULL)
 *   sum_d2      out: sum of dist2 in 64 bits (may be NULL); cloud distance = sqrt(sum_d2) cm
 */
int knn1_nearest(knn1_ctx *ctx, const int16_t *pc_queries, int32_t n_query, const int16_t *pc_ref, int32_t n_ref,
                 int32_t *dist2, int32_t *idx, int64_t *sum_d2);

/*
 * Convert float metres to the planar int16 centimetre format: clip to +/-120 m, multiply by
 * 100, round half to even. xyz holds n points, `stride` floats apart (3 for x,y,z; 4 for KITTI
 * x,y,z,reflectance). planar_out receives 3 * n values. Returns the number of clipped
 * coordinates, or -1 on bad arguments.
 */
int32_t knn1_quantize_cm(const float *xyz, int32_t n, int32_t stride, int16_t *planar_out);

/* Buckets the PVA fetched in the last knn1_nearest() call (a measure of work); -1 if none. */
int64_t knn1_last_buckets_fetched(const knn1_ctx *ctx);

/* Text for an error code. */
const char *knn1_error_string(int code);

#ifdef __cplusplus
}
#endif

#endif /* KNN1_DISTANCE_H */
