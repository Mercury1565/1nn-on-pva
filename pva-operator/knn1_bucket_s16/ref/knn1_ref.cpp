#include "knn1_ref.hpp"

namespace pva_knn_ref {

int64_t dist2(const int16_t *q, int32_t nq, int32_t qi, const int16_t *r, int32_t nr, int32_t ri)
{
    int64_t s = 0;
    for (int32_t a = 0; a < 3; a++)
    {
        const int64_t d = static_cast<int64_t>(q[a * nq + qi]) - static_cast<int64_t>(r[a * nr + ri]);
        s += d * d;
    }
    return s;
}

void nearest(const int16_t *q, int32_t nq, const int16_t *r, int32_t nr, int64_t *outDist2, int32_t *outIdx)
{
    for (int32_t i = 0; i < nq; i++)
    {
        int64_t best  = INT64_MAX;
        int32_t bestJ = -1;
        for (int32_t j = 0; j < nr; j++)
        {
            const int64_t d = dist2(q, nq, i, r, nr, j);
            if (d < best)
            {
                best  = d;
                bestJ = j;
            }
        }
        outDist2[i] = best;
        outIdx[i]   = bestJ;
    }
}

} // namespace pva_knn_ref
