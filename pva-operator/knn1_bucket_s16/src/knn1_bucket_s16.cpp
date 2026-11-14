#include "knn1_bucket_s16.hpp"

#include "../vpu/knn1_bucket_s16_params.h"
#include "bucket_prep.hpp"
#include "knn1_diag.hpp"

#include <cupva_platform.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>

PVA_DECLARE_EXECUTABLE(knn1_bucket_s16_dev)

namespace pva_knn {

static_assert(Knn1BucketS16::kMaxPoints == K1_MAX_POINTS, "size mismatch");
static_assert(Knn1BucketS16::kBucketSize == K1_BUCKET, "size mismatch");
static_assert(Knn1BucketS16::kTeamSize == K1_TEAM, "size mismatch");

namespace {

/* Error codes (any nonzero value means failure). */
constexpr int32_t kErrState     = 1; /* call out of order */
constexpr int32_t kErrArgument  = 2; /* null pointer or size out of range */
constexpr int32_t kErrCupva     = 3; /* cuPVA exception */
constexpr int32_t kErrOther     = 4; /* any other exception */
constexpr int32_t kErrBadResult = 5; /* PVA output inconsistent (bad slot or lane-order check failed) */

/* One cuPVA allocation and its host-visible pointer. */
template <typename T>
struct DeviceBuffer
{
    T *dev{nullptr};
    T *host{nullptr};

    void alloc(size_t count)
    {
        dev  = static_cast<T *>(cupva::mem::Alloc(static_cast<int64_t>(count * sizeof(T))));
        host = static_cast<T *>(cupva::mem::GetHostPointer(dev));
    }
    void release() noexcept
    {
        if (dev != nullptr)
        {
            try
            {
                cupva::mem::Free(dev);
            }
            catch (...)
            {
            }
            dev = host = nullptr;
        }
    }
};

} // namespace

struct Knn1BucketS16::Impl
{
    DeviceBuffer<int16_t> buckets; /* [K1_MAX_BUCKETS][K1_BUCKET_ROW]                        */
    DeviceBuffer<int16_t> teams;   /* [K1_MAX_CHUNKS + K1_BOX_CHUNK_ROWS][K1_CHUNK_IN_W] (+ box table) */
    DeviceBuffer<int32_t> results; /* [K1_MAX_CHUNKS + K1_MAX_VPUS][K1_CHUNK_OUT_W] (+ statistics) */

    /* Statistics of VPU program v (host view). */
    const int32_t *stats(int32_t v) const
    {
        return results.host + static_cast<size_t>(K1_MAX_CHUNKS + v) * K1_CHUNK_OUT_W;
    }
    int16_t *boxTable() const
    {
        return teams.host + static_cast<size_t>(K1_MAX_CHUNKS) * K1_CHUNK_IN_W;
    }

    /* One program per VPU, each for its own range of teams (same executable). */
    int32_t vpus{1};
    int32_t cpuThreads{1}; /* CPU threads for prepare() */
    cupva::Executable exec;
    cupva::CmdProgram prog[K1_MAX_VPUS];
    cupva::CmdBarrier barrier;
    cupva::CmdStatus status[K1_MAX_VPUS + 2]{};

    prep::Bookkeeping book;
    bool prepared{false};
    bool submitted{false};

    ~Impl()
    {
        buckets.release();
        teams.release();
        results.release();
    }
};

Knn1BucketS16::Knn1BucketS16() noexcept = default;
Knn1BucketS16::~Knn1BucketS16()         = default;

int32_t Knn1BucketS16::init(int32_t vpus, int32_t cpu_threads) noexcept
{
    if (vpus < 1 || vpus > K1_MAX_VPUS || cpu_threads < 1 || cpu_threads > kMaxCpuThreads)
    {
        return kErrArgument;
    }
    try
    {
        auto impl  = std::make_unique<Impl>();
        impl->vpus       = vpus;
        impl->cpuThreads = cpu_threads;
        impl->buckets.alloc(static_cast<size_t>(K1_MAX_BUCKETS) * K1_BUCKET_ROW);
        impl->teams.alloc(static_cast<size_t>(K1_MAX_CHUNKS + K1_BOX_CHUNK_ROWS) * K1_CHUNK_IN_W);
        impl->results.alloc(static_cast<size_t>(K1_MAX_CHUNKS + K1_MAX_VPUS) * K1_CHUNK_OUT_W);
        impl->exec = cupva::Executable::Create(PVA_EXECUTABLE_DATA(knn1_bucket_s16_dev),
                                               PVA_EXECUTABLE_SIZE(knn1_bucket_s16_dev));
        for (int32_t v = 0; v < vpus; v++)
        {
            impl->prog[v] = cupva::CmdProgram::Create(impl->exec);
            cupva::CmdProgram &p = impl->prog[v];

            /* Team chunks (+ box table) in: the VPU picks the row at run time. */
            p.addDataFlowHead<cupva::GatherScatterDataFlow>()
                .handler(p["teamHdl"])
                .numTilesPerTrigger(1)
                .tile(K1_CHUNK_IN_W, 1)
                .src(impl->teams.dev, K1_CHUNK_IN_W, K1_MAX_CHUNKS + K1_BOX_CHUNK_ROWS, K1_CHUNK_IN_W);

            /* One bucket per trigger: x, y, z tiles into one cache slot in B, C, D. */
            p.addDataFlowHead<cupva::GatherScatterDataFlow>()
                .handler(p["bucketHdl"])
                .numTilesPerTrigger(3)
                .tile(K1_BUCKET, 1)
                .src(impl->buckets.dev, K1_BUCKET_ROW, K1_MAX_BUCKETS, K1_BUCKET_ROW);

            /* Result chunks (+ statistics) out. */
            p.addDataFlowHead<cupva::GatherScatterDataFlow>()
                .handler(p["resultHdl"])
                .numTilesPerTrigger(1)
                .tile(K1_CHUNK_OUT_W, 1)
                .dst(impl->results.dev, K1_CHUNK_OUT_W, K1_MAX_CHUNKS + K1_MAX_VPUS, K1_CHUNK_OUT_W);

            p.compileDataFlows();
        }
        m_impl = std::move(impl);
        return 0;
    }
    catch (cupva::Exception const &e)
    {
        if (std::getenv("KNN1_DEBUG") != nullptr)
        {
            std::fprintf(stderr, "knn1 init: %s\n", e.what());
        }
        return kErrCupva;
    }
    catch (...)
    {
        return kErrOther;
    }
}

int32_t Knn1BucketS16::prepare(const int16_t *pc_queries, int32_t n_query, const int16_t *pc_ref, int32_t n_ref) noexcept
{
    if (!m_impl)
    {
        return kErrState;
    }
    if (pc_queries == nullptr || pc_ref == nullptr || n_query < 1 || n_query > kMaxPoints || n_ref < 1 ||
        n_ref > kMaxPoints)
    {
        return kErrArgument;
    }
    try
    {
        Impl &m = *m_impl;
        m.prepared = m.submitted = false;
        prep::prepare(pc_queries, n_query, pc_ref, n_ref, prep::DeviceLayout{m.buckets.host, m.boxTable(), m.teams.host},
                      m.book, m.cpuThreads);
        /* Sizes for the VPU programs; each VPU gets a contiguous half of the chunks, so its cache
         * holds only its half of the scene (alternating chunks was tried: more DMA, not faster). */
        const int32_t chunks = (m.book.nTeams + K1_TEAMS_PER_CHUNK - 1) / K1_TEAMS_PER_CHUNK;
        for (int32_t v = 0; v < m.vpus; v++)
        {
            const int32_t c0           = chunks * v / m.vpus;
            const int32_t c1           = chunks * (v + 1) / m.vpus;
            m.prog[v]["nBuckets"]      = m.book.nBuckets;
            m.prog[v]["nBucketGroups"] = m.book.nBucketGroups;
            m.prog[v]["teamBegin"]     = c0 * K1_TEAMS_PER_CHUNK;
            m.prog[v]["teamEnd"]       = std::min(c1 * K1_TEAMS_PER_CHUNK, m.book.nTeams);
            m.prog[v]["vpuIndex"]      = v;
        }
        m.prepared = true;
        return 0;
    }
    catch (cupva::Exception const &)
    {
        return kErrCupva;
    }
    catch (...)
    {
        return kErrOther;
    }
}

int32_t Knn1BucketS16::submit(cupva::Stream &stream, cupva::Fence &fence) noexcept
{
    if (!m_impl || !m_impl->prepared)
    {
        return kErrState;
    }
    try
    {
        Impl &m = *m_impl;
        cupva::CmdRequestFences rf{fence};
        if (m.vpus == 1)
        {
            stream.submit({&m.prog[0], &rf}, m.status);
        }
        else
        {
            /* Both VPUs run side by side; the barrier holds the fence until both finish. */
            stream.submit({&m.prog[0], &m.prog[1], &m.barrier, &rf}, m.status, cupva::OUT_OF_ORDER);
        }
        m_impl->submitted = true;
        return 0;
    }
    catch (cupva::Exception const &)
    {
        return kErrCupva;
    }
    catch (...)
    {
        return kErrOther;
    }
}

int32_t Knn1BucketS16::results(int32_t *dist2, int32_t *idx) noexcept
{
    if (!m_impl || !m_impl->submitted)
    {
        return kErrState;
    }
    if (dist2 == nullptr || idx == nullptr)
    {
        return kErrArgument;
    }
    Impl const &m              = *m_impl;
    const prep::Bookkeeping &b = m.book;
    for (int32_t v = 0; v < m.vpus; v++)
    {
        if (m.stats(v)[K1_STAT_LANE_ORDER_OK] != 1)
        {
            return kErrBadResult;
        }
    }
    const int32_t maxSlot = b.nBucketGroups * K1_GROUP * K1_BUCKET;

    /* Check first, then write, so a bad result leaves the caller's buffers untouched. */
    for (int32_t pass = 0; pass < 2; pass++)
    {
        for (int32_t team = 0; team < b.nTeams; team++)
        {
            /* The VPU writes each team's 16 answers in VSumSq slot order (see the params header). */
            const int32_t *row = m.results.host + (team / K1_TEAMS_PER_CHUNK) * K1_CHUNK_OUT_W +
                                 (team % K1_TEAMS_PER_CHUNK) * K1_RESULT_ROW;
            for (int32_t s = 0; s < K1_TEAM; s++)
            {
                const int32_t i = team * K1_TEAM + K1_LANE_OF_SLOT(s); /* index into queryOfLane */
                if (i >= b.nQuery)
                {
                    continue; /* padding copy of a query */
                }
                const int32_t d2   = row[s];
                const int32_t slot = row[K1_TEAM + s];
                if (pass == 0)
                {
                    if (slot < 0 || slot >= maxSlot || d2 < 0)
                    {
                        return kErrBadResult;
                    }
                    continue;
                }
                const int32_t q = b.queryOfLane[i];
                dist2[q]        = d2;
                idx[q]          = b.refOfSlot[slot];
            }
        }
    }
    return 0;
}

int32_t Knn1BucketS16::stats(Knn1Stats *out) noexcept
{
    if (!m_impl || !m_impl->submitted)
    {
        return kErrState;
    }
    if (out == nullptr)
    {
        return kErrArgument;
    }
    const prep::Bookkeeping &b = m_impl->book;
    out->n_query               = b.nQuery;
    out->n_ref                 = b.nRef;
    out->n_buckets             = b.nBuckets;
    out->n_teams               = b.nTeams;
    out->buckets_fetched       = 0;
    out->dma_fetches           = 0;
    for (int32_t v = 0; v < m_impl->vpus; v++)
    {
        out->buckets_fetched += m_impl->stats(v)[K1_STAT_BUCKETS_FETCHED];
        out->dma_fetches += m_impl->stats(v)[K1_STAT_DMA_FETCHES];
    }
    return 0;
}

int32_t Knn1BucketS16::vpus() const noexcept
{
    return m_impl ? m_impl->vpus : 0;
}

cupva::Error Knn1BucketS16::lastStatus() const noexcept
{
    if (!m_impl)
    {
        return cupva::Error::InvalidState;
    }
    for (int32_t v = 0; v < m_impl->vpus; v++)
    {
        const cupva::Error e = cupva::CheckCommandStatus(m_impl->status[v]);
        if (e != cupva::Error::None)
        {
            return e;
        }
    }
    return cupva::Error::None;
}

int32_t Knn1Diag::rawStats(Knn1BucketS16 const &op, int32_t vpu, uint32_t out[16]) noexcept
{
    if (!op.m_impl || !op.m_impl->submitted || out == nullptr || vpu < 0 || vpu >= op.m_impl->vpus)
    {
        return kErrState;
    }
    for (int32_t i = 0; i < K1_STATS_LEN; i++)
    {
        out[i] = static_cast<uint32_t>(op.m_impl->stats(vpu)[i]);
    }
    return 0;
}

int64_t Knn1BucketS16::sumDist2(const int32_t *dist2, int32_t n) noexcept
{
    int64_t s = 0;
    for (int32_t i = 0; dist2 != nullptr && i < n; i++)
    {
        s += dist2[i];
    }
    return s;
}

} // namespace pva_knn
