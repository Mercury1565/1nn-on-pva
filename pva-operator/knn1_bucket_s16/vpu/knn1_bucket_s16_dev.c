/* Knn1BucketS16 VPU program (Thor GEN3): Part B, Steps 5-9, for this VPU's range of teams. */
#include "knn1_bucket_s16_params.h"

#include <cupva_device.h>


/* Diagnostic builds (-DK1_DIAG_TSC): VPU cycles per phase, reported in stats slots 4-14.
 * The counter pauses while the VPU sleeps on DMA unless built with PVA_PROFILING_MODE=ON. */
#ifdef K1_DIAG_TSC
VMEM(A, uint32_t, diagCycles, K1_STATS_LEN);
#define DIAG_BEGIN(t)                                                                                                  \
    const uint32_t t = read_TSCL();                                                                                    \
    chess_separator_scheduler()
#define DIAG_END(slot, t)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        chess_separator_scheduler();                                                                                   \
        diagCycles[slot] += read_TSCL() - (t);                                                                         \
    } while (0)
#else
#define DIAG_BEGIN(t)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define DIAG_END(slot, t)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#endif

/* ---- VMEM buffers (docs/knn1_bucket_s16_vmem.md) ---- */

/* Bucket cache: x in superbank B, y in C, z in D, so the scan loads all three in one cycle.
 * Each plane ends with two rows of the box table. */
VMEM(B, int16_t, planeB, K1_PLANE_ELEMS);
VMEM(C, int16_t, planeC, K1_PLANE_ELEMS);
VMEM(D, int16_t, planeD, K1_PLANE_ELEMS);
#define cacheX planeB
#define cacheY planeC
#define cacheZ planeD
#define K1_BOX_TAIL (K1_CACHE_SLOTS * K1_BUCKET)
#define boxLoX (planeB + K1_BOX_TAIL)
#define boxHiY (planeB + K1_BOX_TAIL + K1_MAX_BUCKETS)
#define boxHiX (planeC + K1_BOX_TAIL)
#define boxLoZ (planeC + K1_BOX_TAIL + K1_MAX_BUCKETS)
#define boxLoY (planeD + K1_BOX_TAIL)
#define boxHiZ (planeD + K1_BOX_TAIL + K1_MAX_BUCKETS)

/* Double-buffered team chunks in and result chunks out. */
VMEM(A, int16_t, teamBuf, 2 * K1_CHUNK_IN_W);
VMEM_GSDF_HANDLER(A, teamHdl, 1);
VMEM(A, int32_t, resultBuf, 2 * K1_CHUNK_OUT_W);
VMEM_GSDF_HANDLER(A, resultHdl, 1);

/* One bucket row -> one cache slot (3 tiles: x, y, z). */
VMEM_GSDF_HANDLER(A, bucketHdl, 3);

/* Set by the host before each run. */
VMEM(A, int32_t, nBuckets);      /* real buckets */
VMEM(A, int32_t, nBucketGroups); /* groups of 16 buckets */
VMEM(A, int32_t, teamBegin);     /* this VPU's first team */
VMEM(A, int32_t, teamEnd);       /* one past its last team */
VMEM(A, int32_t, vpuIndex);      /* statistics row */

/* Ranking state of the current team. */
VMEM(A, int32_t, rankKeys, K1_MAX_BUCKETS);  /* bucket keys, slot order per group */
VMEM(A, int32_t, groupKey, K1_MAX_GROUPS);   /* group bound key, or its smallest bucket key once expanded */
VMEM(A, int16_t, expandedBy, K1_MAX_GROUPS); /* team that expanded the group (-1: none) */
VMEM(A, int32_t, groupLaneIdsLo, 8);         /* 0, 2, ..., 14 */
VMEM(A, int32_t, groupLaneIdsHi, 8);         /* 1, 3, ..., 15 */
/* Box around each group's 16 bucket boxes. */
VMEM(A, int16_t, groupLoX, K1_MAX_GROUPS);
VMEM(A, int16_t, groupLoY, K1_MAX_GROUPS);
VMEM(A, int16_t, groupLoZ, K1_MAX_GROUPS);
VMEM(A, int16_t, groupHiX, K1_MAX_GROUPS);
VMEM(A, int16_t, groupHiY, K1_MAX_GROUPS);
VMEM(A, int16_t, groupHiZ, K1_MAX_GROUPS);
VMEM(A, int16_t, laneIds, K1_TEAM); /* lane-order self-check */
VMEM(A, int32_t, tmpA, K1_TEAM);

/* Cache maps (-1 = empty). */
VMEM(A, int16_t, slotOfBucket, K1_MAX_BUCKETS);
VMEM(A, int16_t, bucketOfSlot, K1_CACHE_SLOTS);

/* ---- Helpers ---- */

/* VSumSq puts even lanes in the low half, odd lanes in the high half (Thor guide 6.2.3.6). */
static inline int32_t slotOfLane(int32_t l)
{
    return (l & 1) ? 8 + (l >> 1) : (l >> 1);
}

/* 16 lanes -> 16 int32 in slot order. */
static inline void store16(int32_t *dst, dvintx v)
{
    *(vint *)dst       = extract(v.lo);
    *(vint *)(dst + 8) = extract(v.hi);
}

static inline dvintx splat16(int32_t value)
{
    dvintx v;
    v.lo = replicatew(value);
    v.hi = v.lo;
    return v;
}

/* Squared distance from each of 16 points to one box (0 inside). */
static inline dvintx lowerBoundToBox(vshortx px, vshortx py, vshortx pz, int32_t loX, int32_t loY, int32_t loZ,
                                     int32_t hiX, int32_t hiY, int32_t hiZ, vshortx zero)
{
    vshortx gx = vmax(px - hiX, 0) - vmin(px - loX, 0);
    vshortx gy = vmax(py - hiY, 0) - vmin(py - loY, 0);
    vshortx gz = vmax(pz - hiZ, 0) - vmin(pz - loZ, 0);
    return vsumsq(gx, gy) + vsumsq(gz, zero);
}

/* ---- Bucket cache: oldest slot reused first, one DMA in flight ---- */
static int32_t clockHand;   /* next slot to reuse */
static int32_t pendingSlot; /* slot being filled, -1 if none */
static int32_t dmaFetches;

static void cacheReset(void)
{
    for (int32_t b = 0; b < K1_MAX_BUCKETS; b++)
    {
        slotOfBucket[b] = -1;
    }
    for (int32_t s = 0; s < K1_CACHE_SLOTS; s++)
    {
        bucketOfSlot[s] = -1;
    }
    clockHand   = 0;
    pendingSlot = -1;
    dmaFetches  = 0;
}

static int32_t planeStride; /* elements from plane B to C (= C to D) */

static inline void waitPending(void)
{
    if (pendingSlot >= 0)
    {
        cupvaGSDFSync(bucketHdl);
        pendingSlot = -1;
    }
}

/* Copy bucket b into the oldest slot other than `busy`; returns the slot. */
static int32_t startFetch(int32_t b, int32_t busy)
{
    waitPending(); /* GSDF: one transfer at a time */
    int32_t s = clockHand;
    if (s == busy)
    {
        s = (s + 1 == K1_CACHE_SLOTS) ? 0 : s + 1;
    }
    clockHand = (s + 1 == K1_CACHE_SLOTS) ? 0 : s + 1;

    const int32_t old = bucketOfSlot[s];
    if (old >= 0)
    {
        slotOfBucket[old] = -1;
    }
    bucketOfSlot[s] = (int16_t)b;
    slotOfBucket[b] = (int16_t)s;

    int32_t x[3] = {0, K1_BUCKET, 2 * K1_BUCKET}; /* x, y, z parts of row b */
    int32_t y[3] = {b, b, b};
    cupvaGSDFUpdateVmemAddr(bucketHdl, cacheX + s * K1_BUCKET, K1_BUCKET, planeStride);
    cupvaGSDFUpdateTiles(bucketHdl, x, y, 3);
    cupvaGSDFTrig(bucketHdl);
    pendingSlot = s;
    dmaFetches++;
    return s;
}

/* Slot holding bucket b, ready to read (fetches on a miss). */
static inline int32_t bucketSlot(int32_t b)
{
    int32_t s = slotOfBucket[b];
    if (s < 0)
    {
        s = startFetch(b, -1);
    }
    if (s == pendingSlot)
    {
        waitPending();
    }
    return s;
}

/* ---- Step 6: rank buckets by distance from the team's box ----
 * key = (bound, low K1_KEY_ID_BITS cleared) | bucket id. Groups of 16 buckets are ranked first;
 * a group's bucket keys are computed only when the group comes up. A group's bound is <= its
 * buckets' bounds, so the visiting order matches the model exactly. */

typedef struct
{
    int32_t loX, loY, loZ, hiX, hiY, hiZ;
} TeamBox;

/* Keys of 16 boxes (index i0..i0+15) against the team box. */
static inline dvintx boxKeys(const int16_t *loXs, const int16_t *loYs, const int16_t *loZs, const int16_t *hiXs,
                             const int16_t *hiYs, const int16_t *hiZs, int32_t i0, const TeamBox *t, dvintx ids,
                             vshortx zero)
{
    const vshortx loX = sign_extend(*(vshort *)(loXs + i0));
    const vshortx loY = sign_extend(*(vshort *)(loYs + i0));
    const vshortx loZ = sign_extend(*(vshort *)(loZs + i0));
    const vshortx hiX = sign_extend(*(vshort *)(hiXs + i0));
    const vshortx hiY = sign_extend(*(vshort *)(hiYs + i0));
    const vshortx hiZ = sign_extend(*(vshort *)(hiZs + i0));

    /* Gap per axis: box above team, team above box, or overlap (0). */
    const vshortx gx    = vmax(vmax(loX - t->hiX, replicateh(t->loX) - hiX), zero);
    const vshortx gy    = vmax(vmax(loY - t->hiY, replicateh(t->loY) - hiY), zero);
    const vshortx gz    = vmax(vmax(loZ - t->hiZ, replicateh(t->loZ) - hiZ), zero);
    const dvintx bound  = vsumsq(gx, gy) + vsumsq(gz, zero);
    return ((bound >> K1_KEY_ID_BITS) << K1_KEY_ID_BITS) + ids;
}

/* Group keys, 16 groups per vector; unused entries of the last vector = infinity. */
static void rankGroups(const TeamBox *t, int32_t groups, vshortx zero)
{
    dvintx laneIds;
    laneIds.lo = sign_extend(*(vint *)groupLaneIdsLo);
    laneIds.hi = sign_extend(*(vint *)groupLaneIdsHi);
    const dvintx firstBucketIds = laneIds << 4; /* group g starts at bucket 16 g */

    for (int32_t g0 = 0; g0 < groups; g0 += K1_GROUP)
    {
        const dvintx key = boxKeys(groupLoX, groupLoY, groupLoZ, groupHiX, groupHiY, groupHiZ, g0, t,
                                   firstBucketIds + g0 * K1_GROUP, zero);
        store16(groupKey + g0, key);
    }
    for (int32_t g = groups; g < ((groups + K1_GROUP - 1) & ~(K1_GROUP - 1)); g++)
    {
        groupKey[(g & ~(K1_GROUP - 1)) + slotOfLane(g & (K1_GROUP - 1))] = K1_INF32;
    }
}

/* Position of entry i in a slot-ordered table. */
static inline int32_t keyPosition(int32_t i)
{
    return (i & ~(K1_GROUP - 1)) + slotOfLane(i & (K1_GROUP - 1));
}

/* groupKey = smallest remaining bucket key of group g. */
static inline void updateGroupKey(int32_t g)
{
    const int32_t *k        = rankKeys + g * K1_GROUP;
    groupKey[keyPosition(g)] = vminr_s(vmin(sign_extend(*(vint *)k), sign_extend(*(vint *)(k + 8))));
}

/* Compute the 16 bucket keys of group g; padding buckets get infinity. */
static void expandGroup(int32_t g, const TeamBox *t, int32_t buckets, vshortx zero)
{
    dvintx laneIds;
    laneIds.lo   = sign_extend(*(vint *)groupLaneIdsLo);
    laneIds.hi   = sign_extend(*(vint *)groupLaneIdsHi);
    const int32_t b0 = g * K1_GROUP;
    store16(rankKeys + b0, boxKeys(boxLoX, boxLoY, boxLoZ, boxHiX, boxHiY, boxHiZ, b0, t, laneIds + b0, zero));
    for (int32_t b = buckets; b < b0 + K1_GROUP; b++)
    {
        rankKeys[keyPosition(b)] = K1_INF32;
    }
    updateGroupKey(g);
}

/* Smallest key over all groups. */
static inline int32_t smallestKey(int32_t groupSlots)
{
    vintx m = sign_extend(*(vint *)groupKey);
    for (int32_t i = 8; i < groupSlots; i += 8)
    {
        m = vmin(m, sign_extend(*(vint *)(groupKey + i)));
    }
    return vminr_s(m);
}

/* ---- Step 8 scan: closest ref point of bucket slot s for each of the 16 lanes ----
 * Returns (d2 << K1_SLOT_BITS) + j per lane, first j on ties. 8 vector ops per ref point;
 * unroll 4 measured fastest (5.4 cycles per point). */
#ifndef K1_SCAN_UNROLL
#define K1_SCAN_UNROLL 4
#endif
static inline dvintx scanBucket(int32_t s, vshortx qx, vshortx qy, vshortx qz, vshortx zero, dvintx big)
{
    /* Address generators broadcast one ref coordinate to all lanes per load. */
    agen ax = init(cacheX + s * K1_BUCKET);
    agen ay = init(cacheY + s * K1_BUCKET);
    agen az = init(cacheZ + s * K1_BUCKET);
    ax.n1   = K1_BUCKET;
    ay.n1   = K1_BUCKET;
    az.n1   = K1_BUCKET;
    ax.mod1 = sizeof(int16_t);
    ay.mod1 = sizeof(int16_t);
    az.mod1 = sizeof(int16_t);

    dvintx bestD = big >> K1_SLOT_BITS; /* best d2 per lane */
    dvintx bestJ = big;                 /* its position */
    for (int32_t j = 0; j < K1_BUCKET; j++)
        chess_loop_range(K1_BUCKET, K1_BUCKET) chess_unroll_loop(K1_SCAN_UNROLL) chess_prepare_for_pipelining
        {
            const vshortx dx = qx - vshort_load_hs(ax);
            const vshortx dy = qy - vshort_load_hs(ay);
            const vshortx dz = qz - vshort_load_hs(az);
            const dvintx d2  = vsumsq(dx, dy) + vsumsq(dz, zero);
            dvintx closer;
            vminLT(d2.lo, bestD.lo, bestD.lo, closer.lo); /* new min + "strictly closer" flag */
            vminLT(d2.hi, bestD.hi, bestD.hi, closer.hi);
            bestJ = dvmux(closer, j, bestJ);
        }
    return (bestD << K1_SLOT_BITS) + bestJ;
}

/* ---- Steps 5-9 for one team ---- */
static void processTeam(int32_t teamNo, const int16_t *team, int32_t *out, int32_t buckets, int32_t groups,
                        vshortx zero, int32_t *scanned, int32_t *picked)
{
    /* Step 5: load the team, one query point per lane. */
    DIAG_BEGIN(tRank);
    const vshortx qx = sign_extend(*(vshort *)(team));
    const vshortx qy = sign_extend(*(vshort *)(team + K1_TEAM));
    const vshortx qz = sign_extend(*(vshort *)(team + 2 * K1_TEAM));

    /* Step 6: team box, then rank the groups. */
    TeamBox box;
    box.loX = vminr_s(qx);
    box.loY = vminr_s(qy);
    box.loZ = vminr_s(qz);
    box.hiX = vmaxr_s(qx);
    box.hiY = vmaxr_s(qy);
    box.hiZ = vmaxr_s(qz);
    rankGroups(&box, groups, zero);
    const int32_t groupSlots = (groups + K1_GROUP - 1) & ~(K1_GROUP - 1);

    /* Step 7: best per lane = infinity. */
    const dvintx big = splat16(K1_INF32) << K1_SLOT_BITS;
    dvintx bestKey   = big;          /* (d2 << K1_SLOT_BITS) | j */
    dvintx bestB     = splat16(0);   /* bucket of the best */
    int32_t worstBest = K1_INF32;    /* largest best d2 over the lanes */
    int32_t nextKey   = -1;          /* smallest key if already known */
    DIAG_END(K1_DIAG_RANK, tRank);

    /* Step 8: visit buckets nearest first. */
    for (;;)
    {
        /* Pick the smallest key. */
        DIAG_BEGIN(tPick);
        const int32_t key = (nextKey >= 0) ? nextKey : smallestKey(groupSlots);
        nextKey           = -1;
        if (key == K1_INF32)
        {
            DIAG_END(K1_DIAG_PICK, tPick);
            break; /* all visited */
        }
        const int32_t b     = key & K1_KEY_ID_MASK; /* bucket, or first bucket of a group */
        const int32_t g     = b / K1_GROUP;
        const int32_t bound = key & ~K1_KEY_ID_MASK;

        /* Stop: no remaining bucket can beat any lane's best. */
        if (bound >= worstBest)
        {
            (*picked)++;
            DIAG_END(K1_DIAG_PICK, tPick);
            break;
        }

        /* A group came up: compute its bucket keys and pick again. */
        if (expandedBy[g] != teamNo)
        {
            expandGroup(g, &box, buckets, zero);
            expandedBy[g] = (int16_t)teamNo;
            DIAG_END(K1_DIAG_PICK, tPick);
            continue;
        }
        (*picked)++;
        rankKeys[keyPosition(b)] = K1_INF32; /* mark visited */
        updateGroupKey(g);
        DIAG_END(K1_DIAG_PICK, tPick);

        /* Skip: no lane is closer to the box than its current best. */
        DIAG_BEGIN(tSkip);
        const dvintx laneBound =
            lowerBoundToBox(qx, qy, qz, boxLoX[b], boxLoY[b], boxLoZ[b], boxHiX[b], boxHiY[b], boxHiZ[b], zero);
        const dvintx bestD2 = bestKey >> K1_SLOT_BITS;
        const int32_t margin = vminr_s(vmin(laneBound.lo - bestD2.lo, laneBound.hi - bestD2.hi));
        DIAG_END(K1_DIAG_SKIPTEST, tSkip);
        if (margin >= 0)
        {
            continue;
        }

        /* Get the bucket; prefetch the next candidate while scanning. */
        DIAG_BEGIN(tFetch);
        const int32_t s = bucketSlot(b);
        nextKey         = smallestKey(groupSlots);
        if (nextKey != K1_INF32 && (nextKey & ~K1_KEY_ID_MASK) < worstBest)
        {
            const int32_t b2 = nextKey & K1_KEY_ID_MASK;
            if (expandedBy[b2 / K1_GROUP] == teamNo && slotOfBucket[b2] < 0) /* a bucket, not cached */
            {
                startFetch(b2, s);
            }
        }
        DIAG_END(K1_DIAG_FETCH, tFetch);

        /* Scan. */
        DIAG_BEGIN(tScan);
        const dvintx found = scanBucket(s, qx, qy, qz, zero, big);
        DIAG_END(K1_DIAG_SCAN, tScan);
        (*scanned)++;

        /* Merge into the running best. */
        DIAG_BEGIN(tMerge);
        vintx flagLo, flagHi;
        vminLT(found.lo, bestKey.lo, bestKey.lo, flagLo);
        vminLT(found.hi, bestKey.hi, bestKey.hi, flagHi);
        bestB.lo  = vmux(flagLo, b, bestB.lo);
        bestB.hi  = vmux(flagHi, b, bestB.hi);
        worstBest = vmaxr_s(vmax(bestKey.lo, bestKey.hi) >> K1_SLOT_BITS);
        DIAG_END(K1_DIAG_MERGE, tMerge);
    }

    /* Step 9: d2 and ref position (bucket * K1_BUCKET + j), slot order. */
    DIAG_BEGIN(tResult);
    store16(out, bestKey >> K1_SLOT_BITS);
    store16(out + K1_TEAM, (bestB << K1_SLOT_BITS) + (bestKey & (K1_BUCKET - 1)));
    DIAG_END(K1_DIAG_RESULT, tResult);
}

/* Self-check: VSumSq lane order matches the guide. */
static int32_t laneOrderMatchesGuide(vshortx zero)
{
    for (int32_t l = 0; l < K1_TEAM; l++)
    {
        laneIds[l] = (int16_t)l;
    }
    store16(tmpA, vsumsq(sign_extend(*(vshort *)laneIds), zero)); /* slot s = lane^2 */
    int32_t ok = 1;
    for (int32_t s = 0; s < K1_TEAM; s++)
    {
        const int32_t l = K1_LANE_OF_SLOT(s);
        ok &= (tmpA[s] == l * l);
    }
    return ok;
}

/* Start the DMA of row c between DRAM and buf. */
static inline void startChunk(GSDFHandler *hdl, void *buf, int32_t width, int32_t c)
{
    int32_t x = 0;
    int32_t y = c;
    cupvaGSDFUpdateVmemAddr(hdl, buf, width, width);
    cupvaGSDFUpdateTiles(hdl, &x, &y, 1);
    cupvaGSDFTrig(hdl);
}

CUPVA_VPU_MAIN()
{
#ifdef K1_DIAG_TSC
    enable_TSC();
    for (int32_t s = 0; s < K1_STATS_LEN; s++)
    {
        diagCycles[s] = 0; /* VMEM survives between runs */
    }
#endif
    DIAG_BEGIN(tStartup);
    const vshortx zero = replicateh(0);

    for (int32_t s = 0; s < 8; s++)
    {
        groupLaneIdsLo[s] = K1_LANE_OF_SLOT(s);
        groupLaneIdsHi[s] = K1_LANE_OF_SLOT(8 + s);
    }
    cacheReset(); /* buckets change between runs */

    /* Open the dataflows; the three planes must be equally spaced. */
    cupvaGSDFOpen(teamHdl, &teamBuf[0], K1_CHUNK_IN_W, K1_CHUNK_IN_W);
    cupvaGSDFOpen(resultHdl, &resultBuf[0], K1_CHUNK_OUT_W, K1_CHUNK_OUT_W);
    planeStride            = (int32_t)(planeC - planeB);
    const int32_t layoutOk = (planeD - planeC) == planeStride;
    cupvaGSDFOpen(bucketHdl, &planeB[0], K1_BUCKET, planeStride);

    /* Load the box table (rows after the team chunks) into the planes. */
    int16_t *const rows[K1_BOX_ROWS] = {boxLoX, boxLoY, boxLoZ, boxHiX, boxHiY, boxHiZ};
    for (int32_t k = 0; k < K1_BOX_CHUNK_ROWS; k++)
    {
        startChunk(teamHdl, teamBuf, K1_CHUNK_IN_W, K1_MAX_CHUNKS + k);
        cupvaGSDFSync(teamHdl);
        for (int32_t i = 0; i < K1_CHUNK_IN_W; i += 16)
        {
            const int32_t e = k * K1_CHUNK_IN_W + i; /* index into [6][K1_MAX_BUCKETS] */
            if (e < K1_BOX_ROWS * K1_MAX_BUCKETS)
            {
                *(vshort *)(rows[e / K1_MAX_BUCKETS] + e % K1_MAX_BUCKETS) = *(const vshort *)(teamBuf + i);
            }
        }
    }

    /* Group boxes for Step 6. */
    for (int32_t g = 0; g < nBucketGroups; g++)
    {
        const int32_t b0 = g * K1_GROUP;
        groupLoX[g]      = (int16_t)vminr_s(sign_extend(*(vshort *)(boxLoX + b0)));
        groupLoY[g]      = (int16_t)vminr_s(sign_extend(*(vshort *)(boxLoY + b0)));
        groupLoZ[g]      = (int16_t)vminr_s(sign_extend(*(vshort *)(boxLoZ + b0)));
        groupHiX[g]      = (int16_t)vmaxr_s(sign_extend(*(vshort *)(boxHiX + b0)));
        groupHiY[g]      = (int16_t)vmaxr_s(sign_extend(*(vshort *)(boxHiY + b0)));
        groupHiZ[g]      = (int16_t)vmaxr_s(sign_extend(*(vshort *)(boxHiZ + b0)));
    }
    for (int32_t g = 0; g < K1_MAX_GROUPS; g++)
    {
        expandedBy[g] = -1;
    }

    /* Refuse to run if the lane order or plane layout is unexpected. */
    const int32_t laneOrderOk = laneOrderMatchesGuide(zero) && layoutOk;
    const int32_t first       = teamBegin;
    const int32_t end         = teamEnd;
    const int32_t buckets     = nBuckets;
    const int32_t groups      = nBucketGroups;
    int32_t scanned = 0, picked = 0, done = 0;
    DIAG_END(K1_DIAG_STARTUP, tStartup);

    /* Process this VPU's chunks of 32 teams. */
    DIAG_BEGIN(tTotal);
    if (laneOrderOk && end > first)
    {
        const int32_t c0 = first / K1_TEAMS_PER_CHUNK;
        const int32_t c1 = (end + K1_TEAMS_PER_CHUNK - 1) / K1_TEAMS_PER_CHUNK;
        startChunk(teamHdl, teamBuf, K1_CHUNK_IN_W, c0);
        int32_t resultPending = 0;
        for (int32_t c = c0; c < c1; c++)
        {
            const int32_t half = (c - c0) & 1;

            /* Wait for chunk c; start loading c+1 into the other half. */
            DIAG_BEGIN(tChunkIn);
            cupvaGSDFSync(teamHdl);
            if (c + 1 < c1)
            {
                startChunk(teamHdl, teamBuf + (1 - half) * K1_CHUNK_IN_W, K1_CHUNK_IN_W, c + 1);
            }
            DIAG_END(K1_DIAG_CHUNK_IN, tChunkIn);

            const int16_t *teams = teamBuf + half * K1_CHUNK_IN_W;
            int32_t *results     = resultBuf + half * K1_CHUNK_OUT_W;
            const int32_t t0     = c * K1_TEAMS_PER_CHUNK;
            const int32_t count  = (end - t0 < K1_TEAMS_PER_CHUNK) ? end - t0 : K1_TEAMS_PER_CHUNK;
            for (int32_t t = 0; t < count; t++)
            {
                processTeam(t0 + t, teams + t * K1_TEAM_ROW, results + t * K1_RESULT_ROW, buckets, groups, zero,
                            &scanned, &picked);
                done++;
            }

            /* Send chunk c's results (after the previous send finished). */
            DIAG_BEGIN(tChunkOut);
            if (resultPending)
            {
                cupvaGSDFSync(resultHdl);
            }
            startChunk(resultHdl, results, K1_CHUNK_OUT_W, c);
            resultPending = 1;
            DIAG_END(K1_DIAG_CHUNK_OUT, tChunkOut);
        }
        DIAG_BEGIN(tLastOut);
        cupvaGSDFSync(resultHdl);
        waitPending(); /* a prefetch may still be in flight */
        DIAG_END(K1_DIAG_CHUNK_OUT, tLastOut);
    }
    DIAG_END(K1_DIAG_TOTAL, tTotal);

    /* Statistics -> result row K1_MAX_CHUNKS + vpuIndex. */
    int32_t *stats = resultBuf;
    for (int32_t i = 0; i < K1_STATS_LEN; i++)
    {
        stats[i] = 0;
    }
    stats[K1_STAT_BUCKETS_FETCHED] = scanned;
    stats[K1_STAT_BUCKETS_PICKED]  = picked;
    stats[K1_STAT_TEAMS]           = done;
    stats[K1_STAT_LANE_ORDER_OK]   = laneOrderOk;
    stats[K1_STAT_DMA_FETCHES]     = dmaFetches;
#ifdef K1_DIAG_TSC
    for (int32_t s = K1_DIAG_STARTUP; s <= K1_DIAG_TOTAL; s++)
    {
        stats[s] = (int32_t)diagCycles[s];
    }
#endif
    startChunk(resultHdl, resultBuf, K1_CHUNK_OUT_W, K1_MAX_CHUNKS + vpuIndex);
    cupvaGSDFSync(resultHdl);

    cupvaGSDFClose(teamHdl);
    cupvaGSDFClose(resultHdl);
    cupvaGSDFClose(bucketHdl);
    return laneOrderOk ? 0 : 1;
}
