/* Cloud distance between two KITTI scans on the PVA; optional per-point answers in out_dir.
 * usage: knn1_bucket_s16_kitti <pc_ref.bin> <pc_queries.bin> [out_dir] */
#include "knn1_bucket_s16.hpp"

#include <cupva_host.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using pva_knn::Knn1BucketS16;

namespace {

/* One scan as planar int16 centimetres [3][n]. */
struct Cloud
{
    int32_t n{0};
    int32_t clamped{0}; /* coordinates changed by the +/-120 m clip */
    std::vector<int16_t> planar;
};

/* Read float32 x,y,z,r; quantize like quantize_cm in src/knn_data.py. */
bool loadScan(const char *path, Cloud &c)
{
    FILE *f = std::fopen(path, "rb");
    if (f == nullptr)
    {
        std::printf("cannot open %s\n", path);
        return false;
    }
    std::vector<float> raw;
    float buf[4096];
    size_t got;
    while ((got = std::fread(buf, sizeof(float), 4096, f)) > 0)
    {
        raw.insert(raw.end(), buf, buf + got);
    }
    std::fclose(f);
    if (raw.empty() || raw.size() % 4 != 0)
    {
        std::printf("%s: %zu floats is not a whole number of points\n", path, raw.size());
        return false;
    }

    c.n = static_cast<int32_t>(raw.size() / 4);
    c.planar.resize(static_cast<size_t>(3) * c.n);
    for (int32_t i = 0; i < c.n; i++)
    {
        for (int32_t a = 0; a < 3; a++) /* x, y, z; reflectance is ignored */
        {
            float v = raw[static_cast<size_t>(i) * 4 + a];
            if (v < -120.0f || v > 120.0f)
            {
                v = (v < 0.0f) ? -120.0f : 120.0f;
                c.clamped++;
            }
            /* nearbyint uses the default rounding mode: round half to even, like numpy.rint. */
            c.planar[static_cast<size_t>(a) * c.n + i] =
                static_cast<int16_t>(std::nearbyint(static_cast<double>(v) * 100.0));
        }
    }
    return true;
}

bool writeInt32(const std::string &path, const std::vector<int32_t> &v)
{
    FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr)
    {
        return false;
    }
    const size_t n = std::fwrite(v.data(), sizeof(int32_t), v.size(), f);
    std::fclose(f);
    return n == v.size();
}

double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <pc_ref.bin> <pc_queries.bin> [out_dir]\n", argv[0]);
        return 2;
    }
    Cloud pcRef, pcQueries;
    if (!loadScan(argv[1], pcRef) || !loadScan(argv[2], pcQueries))
    {
        return 1;
    }
    if (pcRef.n > Knn1BucketS16::kMaxPoints || pcQueries.n > Knn1BucketS16::kMaxPoints)
    {
        std::printf("a scan has more than %d points\n", Knn1BucketS16::kMaxPoints);
        return 1;
    }
    std::printf("pc_ref     %s: %d points (%d coordinates clamped)\n", argv[1], pcRef.n, pcRef.clamped);
    std::printf("pc_queries %s: %d points (%d coordinates clamped)\n", argv[2], pcQueries.n, pcQueries.clamped);

    try
    {
        Knn1BucketS16 op;
        if (op.init() != 0)
        {
            std::printf("init failed\n");
            return 1;
        }

        auto t0 = std::chrono::steady_clock::now();
        if (op.prepare(pcQueries.planar.data(), pcQueries.n, pcRef.planar.data(), pcRef.n) != 0)
        {
            std::printf("prepare failed\n");
            return 1;
        }
        const double prepareMs = msSince(t0);

        cupva::Stream stream = cupva::Stream::Create();
        cupva::SyncObj sync  = cupva::SyncObj::Create();
        cupva::Fence fence{sync};
        t0 = std::chrono::steady_clock::now();
        if (op.submit(stream, fence) != 0)
        {
            std::printf("submit failed\n");
            return 1;
        }
        fence.wait();
        const double pvaMs = msSince(t0);

        std::vector<int32_t> d2(pcQueries.n), idx(pcQueries.n);
        t0 = std::chrono::steady_clock::now();
        if (op.lastStatus() != cupva::Error::None || op.results(d2.data(), idx.data()) != 0)
        {
            std::printf("PVA run failed\n");
            return 1;
        }
        const double resultsMs = msSince(t0);

        pva_knn::Knn1Stats st{};
        op.stats(&st);
        const int64_t sumD2 = Knn1BucketS16::sumDist2(d2.data(), pcQueries.n);
        const double cloudCm = std::sqrt(static_cast<double>(sumD2));

        std::printf("buckets %d, teams %d, buckets fetched %lld (%.2f per team)\n", st.n_buckets, st.n_teams,
                    static_cast<long long>(st.buckets_fetched),
                    static_cast<double>(st.buckets_fetched) / st.n_teams);
        std::printf("sum d2 = %lld cm^2\n", static_cast<long long>(sumD2));
        std::printf("cloud distance sqrt(sum d2) = %.1f cm = %.3f m\n", cloudCm, cloudCm / 100.0);
        std::printf("mean nearest distance (RMS) = %.2f cm\n", std::sqrt(static_cast<double>(sumD2) / pcQueries.n));
        /* Single-call host times, for orientation only (perf/ measures PVA performance). */
        std::printf("host wall time, single call (not PVA performance): prepare %.1f ms, submit+wait %.1f ms, "
                    "results %.1f ms\n",
                    prepareMs, pvaMs, resultsMs);

        if (argc > 3)
        {
            const std::string dir = argv[3];
            if (!writeInt32(dir + "/dist2.s32", d2) || !writeInt32(dir + "/idx.s32", idx))
            {
                std::printf("cannot write results to %s\n", dir.c_str());
                return 1;
            }
            std::printf("wrote %s/dist2.s32 and %s/idx.s32\n", dir.c_str(), dir.c_str());
        }
        return 0;
    }
    catch (cupva::Exception const &e)
    {
        std::printf("cuPVA exception: %s\n", e.what());
        return 1;
    }
}
