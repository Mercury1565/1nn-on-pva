/* Knn1BucketS16: constants shared by the host and the VPU program.
 * DRAM layouts:
 *   buckets  int16 [K1_MAX_BUCKETS][K1_BUCKET_ROW]   row b = x | y | z of bucket b
 *   teams    int16 [K1_MAX_CHUNKS + K1_BOX_CHUNK_ROWS][K1_CHUNK_IN_W]
 *            32 teams per row (x[16] | y[16] | z[16] each), then the box table [6][K1_MAX_BUCKETS]
 *   results  int32 [K1_MAX_CHUNKS + K1_MAX_VPUS][K1_CHUNK_OUT_W]
 *            32 teams per row (d2[16] | refPos[16] in slot order), then one stats row per VPU */
#ifndef KNN1_BUCKET_S16_PARAMS_H
#define KNN1_BUCKET_S16_PARAMS_H

/* Problem limits. */
#define K1_MAX_POINTS 131072 /* per cloud */
#define K1_TEAM 16           /* query points per team = vector lanes */
#define K1_MAX_TEAMS (K1_MAX_POINTS / K1_TEAM) /* 8192 */

/* Ref points per bucket: 128, 256 or 512 (compile time; the host and VPU must agree). */
#ifndef K1_BUCKET
#define K1_BUCKET 128
#endif
#if K1_BUCKET == 128
#define K1_SLOT_BITS 7 /* log2(K1_BUCKET): bits of a position inside a bucket */
#elif K1_BUCKET == 256
#define K1_SLOT_BITS 8
#elif K1_BUCKET == 512
#define K1_SLOT_BITS 9
#else
#error "K1_BUCKET must be 128, 256 or 512"
#endif
#define K1_MAX_BUCKETS (K1_MAX_POINTS / K1_BUCKET)
#define K1_KEY_ID_BITS (17 - K1_SLOT_BITS) /* log2(K1_MAX_BUCKETS): bits of a bucket id */
#define K1_KEY_ID_MASK ((1 << K1_KEY_ID_BITS) - 1)

/* Buckets are ranked 16 at a time (one bucket per lane): a "bucket group". */
#define K1_GROUP 16
#define K1_MAX_GROUPS (K1_MAX_BUCKETS / K1_GROUP)

/* Row widths of the DRAM surfaces (in elements). */
#define K1_BUCKET_ROW (3 * K1_BUCKET) /* int16 */
#define K1_TEAM_ROW (3 * K1_TEAM)     /* 48 int16 */
#define K1_TEAMS_PER_CHUNK 32
#define K1_MAX_CHUNKS (K1_MAX_TEAMS / K1_TEAMS_PER_CHUNK)   /* 256 */
#define K1_CHUNK_IN_W (K1_TEAMS_PER_CHUNK * K1_TEAM_ROW)    /* 1536 int16 */
#define K1_RESULT_ROW (2 * K1_TEAM)                         /* 32 int32 per team */
#define K1_CHUNK_OUT_W (K1_TEAMS_PER_CHUNK * K1_RESULT_ROW) /* 1024 int32 */
#define K1_BOX_ROWS 6
#define K1_BOX_CHUNK_ROWS ((K1_BOX_ROWS * K1_MAX_BUCKETS + K1_CHUNK_IN_W - 1) / K1_CHUNK_IN_W)

/* Bucket cache: one plane per superbank B/C/D (Thor guide 5.2), cache slots then 2 box rows. */
#define K1_PLANE_ELEMS ((124 * 1024) / 2)
#define K1_CACHE_SLOTS ((K1_PLANE_ELEMS - 2 * K1_MAX_BUCKETS) / K1_BUCKET)

/* Lane <-> slot mapping of VSumSq results (Thor guide 6.2.3.6, PDF p.51). */
#define K1_LANE_OF_SLOT(s) (((s) < 8) ? 2 * (s) : 2 * ((s) - 8) + 1)

/* "Infinity" for 32-bit squared distances and keys. Real d2 <= 3 * 24000^2 = 1728000000. */
#define K1_INF32 0x7FFFFFFF

/* Number of VPU programs (one per VPU) the host may use. */
#define K1_MAX_VPUS 2

/* Statistics written by each VPU program at the end of its task. */
#define K1_STATS_LEN 16
#define K1_STAT_BUCKETS_FETCHED 0 /* buckets scanned (from the cache or DRAM), all teams */
#define K1_STAT_BUCKETS_PICKED 1  /* ranking picks, including the pick that triggered the stop */
#define K1_STAT_TEAMS 2           /* teams processed */
#define K1_STAT_LANE_ORDER_OK 3   /* 1 if the VSumSq lane order matched the documented order */
#define K1_STAT_DMA_FETCHES 15    /* buckets copied from DRAM into the cache (misses + prefetches) */

/* Diagnostic builds only (-DK1_DIAG_TSC): VPU cycles per phase in stats slots 4..14. */
#define K1_DIAG_STARTUP 4    /* open dataflows, load box table, lane-order check */
#define K1_DIAG_RANK 5       /* Steps 5-7: load team, team box, rank buckets, reset bests */
#define K1_DIAG_PICK 6       /* Step 8: find the smallest key, stop test, mark visited */
#define K1_DIAG_SKIPTEST 7   /* Step 8: "can any lane improve?" */
#define K1_DIAG_FETCH 8      /* Step 8: cache lookup, DMA trigger, wait for DMA */
#define K1_DIAG_SCAN 9       /* Step 8: the scan loop over one bucket */
#define K1_DIAG_MERGE 10     /* Step 8: merge into the running best, worst best */
#define K1_DIAG_RESULT 11    /* Step 9: write the team's answers */
#define K1_DIAG_CHUNK_IN 12  /* team chunk DMA in (wait) */
#define K1_DIAG_CHUNK_OUT 13 /* result chunk DMA out (wait) */
#define K1_DIAG_TOTAL 14     /* the whole chunk loop */

#endif /* KNN1_BUCKET_S16_PARAMS_H */
