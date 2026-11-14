/* Runs on the CPU before the PVA starts */
/* Receives: two point clouds pc_queries & pc_ref */
/* Produces: 
    - three tables in DRAM for the PVA to read:
        - buckets of ref points
        - one bounding box per bucket (max & min)
        - teams of query points
    - two lookup lists
        - refOfSlot
        - queryOfLane
*/

#include "bucket_prep.hpp"

#include "../vpu/knn1_bucket_s16_params.h"

#include <algorithm>
#include <cstring>
#include <thread>

namespace pva_knn {
namespace prep {

namespace {

/* Planar points: x/y/z arrays plus each point's original index. */
struct View
{
    int16_t *c[3];
    int32_t *id;
};

/* Spans below this (cm) count in one pass; wider spans use two (high byte, then low byte). */
constexpr int32_t kDirectBins = 4096;

void copyRange(const View &from, const View &to, int32_t b, int32_t n)
{
    for (int32_t a = 0; a < 3; a++)
    {
        std::memcpy(to.c[a] + b, from.c[a] + b, sizeof(int16_t) * static_cast<size_t>(n));
    }
    std::memcpy(to.id + b, from.id + b, sizeof(int32_t) * static_cast<size_t>(n));
}

/* First bin where the running count reaches `need`; adds the bins before it to `below`. */
inline int32_t findBin(const int32_t *count, int32_t need, int32_t &below)
{
    int32_t i = 0;
    while (below + count[i] < need)
    {
        below += count[i++];
    }
    return i;
}

/* Median split of points [b, b + size) from `cur` into `other`, then recurse with the buffers
 * swapped. Finished groups of `leaf` points end up in the main buffer (`curIsMain`). */
void split(const View &cur, const View &other, bool curIsMain, int32_t b, int32_t size, int32_t leaf, int32_t *count,
           int32_t threads)
{
    if (size <= leaf)
    {
        if (!curIsMain)
        {
            copyRange(cur, other, b, size);
        }
        return;
    }

    /* Widest axis (first wins a tie, like numpy.argmax). */
    int32_t lo[3], hi[3];
    for (int32_t a = 0; a < 3; a++)
    {
        const int16_t *v = cur.c[a] + b;
        int16_t l = v[0], h = v[0];
        for (int32_t k = 1; k < size; k++)
        {
            l = std::min(l, v[k]);
            h = std::max(h, v[k]);
        }
        lo[a] = l;
        hi[a] = h;
    }
    int32_t axis = 0;
    for (int32_t a = 1; a < 3; a++)
    {
        if (hi[a] - lo[a] > hi[axis] - lo[axis])
        {
            axis = a;
        }
    }

    /* Left half: half the points rounded up to whole leaves, so only the last group can be partial. */
    const int32_t nLeft = ((size / 2 + leaf - 1) / leaf) * leaf;

    /* Median m = nLeft-th smallest value, found by counting; `below` = how many are smaller. */
    const int16_t *v   = cur.c[axis] + b;
    const int32_t base = lo[axis];
    const int32_t span = hi[axis] - base;
    int32_t below      = 0;
    int32_t m;
    if (span < kDirectBins)
    {
        std::fill(count, count + span + 1, 0);
        for (int32_t k = 0; k < size; k++)
        {
            count[v[k] - base]++;
        }
        m = base + findBin(count, nLeft, below);
    }
    else
    {
        const int32_t highBins = (span >> 8) + 1;
        std::fill(count, count + highBins, 0);
        for (int32_t k = 0; k < size; k++)
        {
            count[(v[k] - base) >> 8]++;
        }
        const int32_t high = findBin(count, nLeft, below);
        std::fill(count, count + 256, 0);
        for (int32_t k = 0; k < size; k++)
        {
            const int32_t off = v[k] - base;
            count[off & 255] += ((off >> 8) == high);
        }
        m = base + (high << 8) + findBin(count, nLeft, below);
    }

    /* Move points to `other`: < m left, > m right, == m left until full (branch-free, order kept). */
    int32_t left = b, right = b + nLeft, equalLeft = nLeft - below;
    for (int32_t k = 0; k < size; k++)
    {
        const int32_t x      = v[k];
        const int32_t equal  = (x == m);
        const int32_t toLeft = (x < m) | (equal & (equalLeft > 0));
        const int32_t d      = toLeft ? left : right;
        equalLeft -= equal & toLeft;
        other.c[0][d] = cur.c[0][b + k];
        other.c[1][d] = cur.c[1][b + k];
        other.c[2][d] = cur.c[2][b + k];
        other.id[d]   = cur.id[b + k];
        left += toLeft;
        right += 1 - toLeft;
    }

    /* Split both halves, in parallel if threads allow. */
    if (threads > 1)
    {
        std::vector<int32_t> countLeft(kDirectBins); /* the left half's thread gets its own counters */
        std::thread t(split, std::cref(other), std::cref(cur), !curIsMain, b, nLeft, leaf, countLeft.data(),
                      threads / 2);
        split(other, cur, !curIsMain, b + nLeft, size - nLeft, leaf, count, threads - threads / 2);
        t.join();
        return;
    }
    split(other, cur, !curIsMain, b, nLeft, leaf, count, 1);
    split(other, cur, !curIsMain, b + nLeft, size - nLeft, leaf, count, 1);
}

} // namespace

void medianSplit(const int16_t *planar, int32_t n, int32_t leaf, SplitPoints &out, int32_t threads)
{
    const size_t un = static_cast<size_t>(n);
    for (int32_t a = 0; a < 3; a++)
    {
        out.c[a].assign(planar + a * un, planar + (a + 1) * un);
        out.scratchC[a].resize(un);
    }
    out.id.resize(un);
    out.scratchId.resize(un);
    for (int32_t i = 0; i < n; i++)
    {
        out.id[i] = i;
    }
    const View main{{out.c[0].data(), out.c[1].data(), out.c[2].data()}, out.id.data()};
    const View scratch{{out.scratchC[0].data(), out.scratchC[1].data(), out.scratchC[2].data()}, out.scratchId.data()};
    std::vector<int32_t> count(kDirectBins);
    split(main, scratch, true, 0, n, leaf, count.data(), threads);
}

void prepare(const int16_t *pc_queries, int32_t nQuery, const int16_t *pc_ref, int32_t nRef, DeviceLayout const &dst,
             Bookkeeping &book, int32_t threads)
{
    book.nQuery        = nQuery;
    book.nRef          = nRef;
    book.nBuckets      = (nRef + K1_BUCKET - 1) / K1_BUCKET;
    book.nBucketGroups = (book.nBuckets + K1_GROUP - 1) / K1_GROUP;
    book.nTeams        = (nQuery + K1_TEAM - 1) / K1_TEAM;
    const int32_t paddedBuckets = book.nBucketGroups * K1_GROUP;

    /* Step 4's split is independent: with threads, start it now. */
    SplitPoints &queries = book.queryScratch;
    std::thread queryThread;
    if (threads > 1)
    {
        queryThread = std::thread(medianSplit, pc_queries, nQuery, K1_TEAM, std::ref(queries), threads / 2);
    }

    /* Step 1: order pc_ref so that every run of K1_BUCKET ref points is one bucket. */
    SplitPoints &refs = book.refScratch;
    medianSplit(pc_ref, nRef, K1_BUCKET, refs, threads - threads / 2);

    /* Steps 2-3: write the buckets and their boxes; refOfSlot maps each slot back to pc_ref.
     * Padding with copies keeps the answer exact; the table is padded to a multiple of 16 buckets. */
    book.refOfSlot.resize(static_cast<size_t>(paddedBuckets) * K1_BUCKET);
    for (int32_t bucket = 0; bucket < paddedBuckets; bucket++)
    {
        /* Past the last real bucket: a copy of the last real bucket. */
        const int32_t first = std::min(bucket, book.nBuckets - 1) * K1_BUCKET; /* first sorted point */
        const int32_t real  = std::min(K1_BUCKET, nRef - first);              /* real points in it */
        int16_t *row        = dst.buckets + bucket * K1_BUCKET_ROW;
        int32_t *ids        = book.refOfSlot.data() + bucket * K1_BUCKET;
        std::memcpy(ids, refs.id.data() + first, sizeof(int32_t) * static_cast<size_t>(real));
        std::fill(ids + real, ids + K1_BUCKET, refs.id[first + real - 1]); /* partial: copies of its last point */
        for (int32_t a = 0; a < 3; a++)
        {
            const int16_t *src = refs.c[a].data() + first;
            int16_t *plane     = row + a * K1_BUCKET;
            std::memcpy(plane, src, sizeof(int16_t) * static_cast<size_t>(real));
            std::fill(plane + real, plane + K1_BUCKET, src[real - 1]);
            const auto mm = std::minmax_element(src, src + real);
            dst.boxes[a * K1_MAX_BUCKETS + bucket]       = *mm.first;  /* rows 0..2: min x, y, z */
            dst.boxes[(3 + a) * K1_MAX_BUCKETS + bucket] = *mm.second; /* rows 3..5: max x, y, z */
        }
    }

    /* Step 4: teams of 16 nearby queries (last one padded), 32 teams per DMA row as x | y | z.
     * queryOfLane maps each lane back to pc_queries. */
    if (threads > 1)
    {
        queryThread.join();
    }
    else
    {
        medianSplit(pc_queries, nQuery, K1_TEAM, queries, 1);
    }
    book.queryOfLane.resize(static_cast<size_t>(book.nTeams) * K1_TEAM);
    for (int32_t i = 0; i < book.nTeams * K1_TEAM; i++)
    {
        const int32_t k       = std::min(i, nQuery - 1);
        book.queryOfLane[i]   = queries.id[k];
        const int32_t team    = i / K1_TEAM;
        const int32_t lane    = i % K1_TEAM;
        int16_t *t            = dst.teams + (team / K1_TEAMS_PER_CHUNK) * K1_CHUNK_IN_W +
                     (team % K1_TEAMS_PER_CHUNK) * K1_TEAM_ROW;
        t[lane]               = queries.c[0][k];
        t[K1_TEAM + lane]     = queries.c[1][k];
        t[2 * K1_TEAM + lane] = queries.c[2][k];
    }
}

} // namespace prep
} // namespace pva_knn
