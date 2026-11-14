/*
 * Minimal Knn1BucketS16 lifecycle: make two small point clouds, init, prepare, submit, wait,
 * read the results, check them against the brute-force CPU reference, print the cloud distance.
 */
#include "knn1_bucket_s16.hpp"
#include "knn1_ref.hpp"

#include <cupva_host.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using pva_knn::Knn1BucketS16;

int main()
{
    /* Two random clouds in [-3000, 3000] cm, planar [3][n]. */
    constexpr int32_t NQ = 2000, NR = 3000;
    std::vector<int16_t> q(3 * NQ), r(3 * NR);
    uint32_t s = 12345u;
    auto rnd   = [&]() { s = s * 1664525u + 1013904223u; return static_cast<int16_t>(int32_t(s >> 8) % 6001 - 3000); };
    for (auto &v : q) v = rnd();
    for (auto &v : r) v = rnd();

    try
    {
        Knn1BucketS16 op;
        if (op.init() != 0 || op.prepare(q.data(), NQ, r.data(), NR) != 0)
        {
            std::printf("init/prepare failed\n");
            return 1;
        }
        cupva::Stream stream = cupva::Stream::Create();
        cupva::SyncObj sync  = cupva::SyncObj::Create();
        cupva::Fence fence{sync};
        if (op.submit(stream, fence) != 0)
        {
            std::printf("submit failed\n");
            return 1;
        }
        fence.wait();

        std::vector<int32_t> d2(NQ), idx(NQ);
        if (op.lastStatus() != cupva::Error::None || op.results(d2.data(), idx.data()) != 0)
        {
            std::printf("PVA run failed\n");
            return 1;
        }

        std::vector<int64_t> refD2(NQ);
        std::vector<int32_t> refIdx(NQ);
        pva_knn_ref::nearest(q.data(), NQ, r.data(), NR, refD2.data(), refIdx.data());
        int32_t bad = 0;
        for (int32_t i = 0; i < NQ; i++) bad += (d2[i] != refD2[i]);

        const double cloudDistanceCm = std::sqrt(static_cast<double>(Knn1BucketS16::sumDist2(d2.data(), NQ)));
        std::printf("knn1_bucket_s16 sample: %s (dist2 mismatches: %d), cloud distance sqrt(sum d2) = %.1f cm\n",
                    bad == 0 ? "PASS" : "FAIL", bad, cloudDistanceCm);
        return bad == 0 ? 0 : 1;
    }
    catch (cupva::Exception const &e)
    {
        std::printf("cuPVA exception: %s\n", e.what());
        return 1;
    }
}
