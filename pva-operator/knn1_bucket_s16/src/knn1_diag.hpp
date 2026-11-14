/* Internal, diagnostics only: raw PVA statistics of the last submission. */
#ifndef KNN1_DIAG_HPP
#define KNN1_DIAG_HPP

#include "knn1_bucket_s16.hpp"

#include <cstdint>

namespace pva_knn {

struct Knn1Diag
{
    /* Copies the 16 statistics words (as uint32) of VPU program `vpu` into out. Returns 0, or
     * nonzero if no submission has completed. */
    static int32_t rawStats(Knn1BucketS16 const &op, int32_t vpu, uint32_t out[16]) noexcept;
};

} // namespace pva_knn

#endif // KNN1_DIAG_HPP
