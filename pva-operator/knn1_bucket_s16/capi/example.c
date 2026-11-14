/* Cloud distance between two KITTI scans with the C interface.
 *   gcc -O2 example.c -I ../include -L ../lib -lknn1_distance -lm -Wl,-rpath,'$ORIGIN/../lib' -o example
 *   ./example frame_t-1.bin frame_t.bin
 */
#include "knn1_distance.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Read a KITTI .bin (float32 x, y, z, reflectance) into planar int16 cm. Returns point count. */
static int32_t load(const char *path, int16_t **planar)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    int32_t n = (int32_t)(bytes / 16);
    float *raw = malloc((size_t)bytes);
    if (!raw || fread(raw, 16, (size_t)n, f) != (size_t)n) { fclose(f); free(raw); return -1; }
    fclose(f);
    *planar = malloc(sizeof(int16_t) * 3 * (size_t)n);
    knn1_quantize_cm(raw, n, 4, *planar);
    free(raw);
    return n;
}

int main(int argc, char **argv)
{
    if (argc != 3) { printf("usage: %s <pc_ref.bin> <pc_queries.bin>\n", argv[0]); return 2; }
    int16_t *ref = NULL, *qry = NULL;
    int32_t nr = load(argv[1], &ref), nq = load(argv[2], &qry);
    if (nr < 1 || nq < 1) { printf("cannot read the scans\n"); return 1; }

    knn1_ctx *ctx = knn1_create();
    if (!ctx) { printf("knn1_create failed\n"); return 1; }
    int32_t *dist2 = malloc(sizeof(int32_t) * (size_t)nq), *idx = malloc(sizeof(int32_t) * (size_t)nq);
    int64_t sum = 0;
    int rc = knn1_nearest(ctx, qry, nq, ref, nr, dist2, idx, &sum);
    if (rc != 0) { printf("knn1_nearest: %s\n", knn1_error_string(rc)); return 1; }

    printf("%d query points, %d ref points: cloud distance %.3f m, first point: d2=%d cm^2 -> ref %d\n", nq, nr,
           sqrt((double)sum) / 100.0, dist2[0], idx[0]);
    knn1_destroy(ctx);
    free(ref); free(qry); free(dist2); free(idx);
    return 0;
}
