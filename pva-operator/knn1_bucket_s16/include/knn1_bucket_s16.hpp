/* Knn1BucketS16: exact 1-NN from each query point of pc_queries to pc_ref on the Thor PVA.
 * Inputs: planar int16 cm [3][n], |coord| <= 12000, 1 <= n <= kMaxPoints. Returns 0 on success.
 *     op.init();                                 // once
 *     op.prepare(pc_queries, nq, pc_ref, nr);    // CPU: buckets and teams
 *     op.submit(stream, fence); fence.wait();    // PVA: the search
 *     op.results(dist2, idx);                    // dist2 exact, idx any nearest on a tie
 */
#ifndef KNN1_BUCKET_S16_HPP
#define KNN1_BUCKET_S16_HPP

#include <cupva_host.hpp>

#include <cstdint>
#include <memory>

namespace pva_knn {

/* Work counters reported by the PVA for the last completed submission. */
struct Knn1Stats
{
    int32_t n_query;
    int32_t n_ref;
    int32_t n_buckets;       /* real buckets, ceil(n_ref / kBucketSize) */
    int32_t n_teams;         /* ceil(n_query / 16) */
    int64_t buckets_fetched; /* buckets scanned, summed over all teams */
    int64_t dma_fetches;     /* buckets copied from DRAM into the PVA's bucket cache */
};

/* Ref points per bucket, fixed when the operator is built (CMake option K1_BUCKET). */
#ifndef K1_BUCKET
#define K1_BUCKET 128
#endif

class Knn1BucketS16
{
public:
    static constexpr int32_t kMaxPoints  = 131072;
    static constexpr int32_t kBucketSize = K1_BUCKET;
    static constexpr int32_t kTeamSize   = 16;
    static constexpr int32_t kMaxCpuThreads = 16;

    Knn1BucketS16() noexcept;
    ~Knn1BucketS16();
    Knn1BucketS16(Knn1BucketS16 const &)            = delete;
    Knn1BucketS16 &operator=(Knn1BucketS16 const &) = delete;

    /* Allocate buffers and create the PVA programs. vpus: 1 or 2.
     * cpu_threads: threads prepare() may use (faster, but busier CPU; results unchanged). */
    int32_t init(int32_t vpus = 2, int32_t cpu_threads = 1) noexcept;

    /* Part A on the CPU. pc_queries: int16[3][n_query], pc_ref: int16[3][n_ref], both planar. */
    int32_t prepare(const int16_t *pc_queries, int32_t n_query, const int16_t *pc_ref, int32_t n_ref) noexcept;

    /* Part B on the PVA, asynchronous: results are valid after `fence` expires. */
    int32_t submit(cupva::Stream &stream, cupva::Fence &fence) noexcept;

    /* After the fence: dist2[n_query] and idx[n_query] in the caller's original query order. */
    int32_t results(int32_t *dist2, int32_t *idx) noexcept;

    /* After the fence: work counters of the last submission. */
    int32_t stats(Knn1Stats *out) noexcept;

    /* Number of VPU programs in use (0 before init). */
    int32_t vpus() const noexcept;

    /* cuPVA status of the last submission (after its fence). */
    cupva::Error lastStatus() const noexcept;

    /* Sum of squared distances in 64 bits (a frame's sum exceeds 2^31). */
    static int64_t sumDist2(const int32_t *dist2, int32_t n) noexcept;

private:
    friend struct Knn1Diag; /* internal: reads the raw counters (src/knn1_diag.hpp) */
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace pva_knn

#endif // KNN1_BUCKET_S16_HPP
