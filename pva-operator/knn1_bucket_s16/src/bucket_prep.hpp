/* Part A (Steps 1-4) on the CPU; same rules as prepare() in src/knn1_bucket_model.py. */
#ifndef KNN1_BUCKET_PREP_HPP
#define KNN1_BUCKET_PREP_HPP

#include <cstdint>
#include <vector>

namespace pva_knn {
namespace prep {

/* The operator's DRAM buffers (host pointers). */
struct DeviceLayout
{
    int16_t *buckets; /* [K1_MAX_BUCKETS][K1_BUCKET_ROW] */
    int16_t *boxes;   /* [K1_BOX_ROWS][K1_MAX_BUCKETS]   */
    int16_t *teams;   /* [K1_MAX_CHUNKS][K1_CHUNK_IN_W]  */
};

/* Split result: planar coordinates and original index; scratch arrays are reused between calls. */
struct SplitPoints
{
    std::vector<int16_t> c[3];
    std::vector<int32_t> id;
    std::vector<int16_t> scratchC[3];
    std::vector<int32_t> scratchId;
};

/* What results() needs to map PVA answers back to the caller's order. */
struct Bookkeeping
{
    int32_t nQuery{0};
    int32_t nRef{0};
    int32_t nBuckets{0};      /* real buckets: ceil(nRef / K1_BUCKET) */
    int32_t nBucketGroups{0}; /* ceil(nBuckets / 16); the box table is padded to 16 * groups */
    int32_t nTeams{0};        /* ceil(nQuery / 16) */
    std::vector<int32_t> refOfSlot;   /* ref slot (bucket * K1_BUCKET + position) -> caller's ref index */
    std::vector<int32_t> queryOfLane; /* team * 16 + lane -> caller's query index (first nQuery entries are real) */
    SplitPoints refScratch;   /* working space of Step 1 */
    SplitPoints queryScratch; /* working space of Step 4 */
};

/* Steps 1/4: reorder so every run of `leaf` points is a compact group (bucket_compare.median_order). */
void medianSplit(const int16_t *planar, int32_t n, int32_t leaf, SplitPoints &out, int32_t threads);

/* Steps 1-4: buckets, boxes and teams into `dst`. Inputs planar int16 [3][n]; result independent of `threads`. */
void prepare(const int16_t *pc_queries, int32_t nQuery, const int16_t *pc_ref, int32_t nRef, DeviceLayout const &dst,
             Bookkeeping &book, int32_t threads);

} // namespace prep
} // namespace pva_knn

#endif // KNN1_BUCKET_PREP_HPP
