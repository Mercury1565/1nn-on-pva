/* PVA time split: profiler times plus per-phase VPU cycles from a -DK1_DIAG_TSC build.
 * usage: knn1_bucket_s16_diag <testdata_dir> <case> <warmup> <samples> <out_csv> [vpus=2] */
#include "knn1_bucket_s16.hpp"
#include "../src/knn1_diag.hpp"
#include "../vpu/knn1_bucket_s16_params.h"

#include <cupva_host.hpp>
#include <cupva_host_utils.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using pva_knn::Knn1BucketS16;

namespace {

struct Phase
{
    int32_t slot;
    const char *name;
};

const Phase kPhases[] = {
    {K1_DIAG_STARTUP, "startup"},     {K1_DIAG_RANK, "rank"},         {K1_DIAG_PICK, "pick"},
    {K1_DIAG_SKIPTEST, "skiptest"},   {K1_DIAG_FETCH, "fetch"},       {K1_DIAG_SCAN, "scan"},
    {K1_DIAG_MERGE, "merge"},         {K1_DIAG_RESULT, "result"},     {K1_DIAG_CHUNK_IN, "chunk_in"},
    {K1_DIAG_CHUNK_OUT, "chunk_out"}, {K1_DIAG_TOTAL, "total"},       {K1_STAT_BUCKETS_FETCHED, "scanned"},
    {K1_STAT_DMA_FETCHES, "dma"},     {K1_STAT_TEAMS, "teams"},
};

std::vector<char> readAll(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

double median(std::vector<double> v)
{
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double percentile(std::vector<double> v, double p)
{
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(p / 100.0 * static_cast<double>(v.size())))];
}

std::vector<double> profilerNs(cupva_utils::ProfilingContext &ctx, cupva_utils::ProfilingBatchStatType t)
{
    auto const &d = ctx.getStatistics(t).getStatsData();
    std::vector<double> v;
    for (int64_t i = 0; d.data != nullptr && i < static_cast<int64_t>(d.count); i++) v.push_back(double(d.data[i]));
    return v;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 6 && argc != 7)
    {
        std::fprintf(stderr, "usage: %s <testdata_dir> <case> <warmup> <samples> <out_csv> [vpus=2]\n", argv[0]);
        return 2;
    }
    const std::string dir = std::string(argv[1]) + "/" + argv[2];
    const int warmup = std::atoi(argv[3]), samples = std::atoi(argv[4]);
    const int32_t vpus = (argc == 7) ? std::atoi(argv[6]) : 2;

    std::vector<char> qr = readAll(dir + "/queries.s16"), rr = readAll(dir + "/refs.s16"),
                      er = readAll(dir + "/dist2.s32");
    if (qr.empty() || rr.empty() || er.size() != qr.size() / 6 * 4)
    {
        std::fprintf(stderr, "cannot read %s\n", dir.c_str());
        return 2;
    }
    const int32_t nq = static_cast<int32_t>(qr.size() / 6), nr = static_cast<int32_t>(rr.size() / 6);
    std::vector<int16_t> q(3 * static_cast<size_t>(nq)), r(3 * static_cast<size_t>(nr));
    std::vector<int32_t> exp(nq), d2(nq), idx(nq);
    std::memcpy(q.data(), qr.data(), qr.size());
    std::memcpy(r.data(), rr.data(), rr.size());
    std::memcpy(exp.data(), er.data(), er.size());
    const int64_t expSum = Knn1BucketS16::sumDist2(exp.data(), nq);

    try
    {
        Knn1BucketS16 op;
        const int32_t rcInit = op.init(vpus);
        const int32_t rcPrep = (rcInit == 0) ? op.prepare(q.data(), nq, r.data(), nr) : -1;
        if (rcInit != 0 || rcPrep != 0)
        {
            std::fprintf(stderr, "init failed (%d) or prepare failed (%d)\n", rcInit, rcPrep);
            return 1;
        }
        cupva_utils::ProfilingContext ctx = cupva_utils::ProfilingContext::Create();
        cupva::Stream stream              = cupva_utils::CreateProfilingStream(ctx);
        cupva::SyncObj sync               = cupva::SyncObj::Create();

        std::vector<std::vector<uint32_t>> counters; /* per sample: vpus x 16 words */
        std::vector<double> wallNs;
        int64_t badRuns = 0;
        auto once = [&](bool keep) -> bool {
            /* prepare() once is enough: the inputs never change, only the PVA part is repeated. */
            cupva::Fence fence{sync};
            const auto t0 = std::chrono::steady_clock::now();
            if (op.submit(stream, fence) != 0) return false;
            fence.wait();
            const auto t1 = std::chrono::steady_clock::now();
            if (op.lastStatus() != cupva::Error::None || op.results(d2.data(), idx.data()) != 0) return false;
            badRuns += (Knn1BucketS16::sumDist2(d2.data(), nq) != expSum);
            std::vector<uint32_t> raw(16 * static_cast<size_t>(vpus));
            for (int32_t v = 0; v < vpus; v++) pva_knn::Knn1Diag::rawStats(op, v, raw.data() + 16 * v);
            if (keep)
            {
                counters.push_back(raw);
                wallNs.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            }
            return true;
        };
        for (int i = 0; i < warmup; i++)
        {
            if (!once(false)) return 1;
        }
        ctx.reset();
        for (int i = 0; i < samples; i++)
        {
            if (!once(true)) return 1;
        }

        /* The first two profiler entries after reset() were invalid in earlier runs; keep them in
         * the CSV but leave them out of the medians below if they are absurd (> 10 s). */
        const auto totNs = profilerNs(ctx, cupva_utils::ProfilingBatchStatType::TOTAL_TIME);
        const auto exeNs = profilerNs(ctx, cupva_utils::ProfilingBatchStatType::EXECUTION_TIME);

        std::ofstream f(argv[5]);
        f << "sample,total_ns,exec_ns,wall_ns";
        for (int32_t v = 0; v < vpus; v++)
            for (auto const &p : kPhases) f << ",v" << v << "_" << p.name;
        f << "\n";
        for (size_t i = 0; i < counters.size(); i++)
        {
            f << i << "," << (i < totNs.size() ? totNs[i] : -1.0) << "," << (i < exeNs.size() ? exeNs[i] : -1.0)
              << "," << wallNs[i];
            for (int32_t v = 0; v < vpus; v++)
                for (auto const &p : kPhases) f << "," << counters[i][16 * v + p.slot];
            f << "\n";
        }

        auto sane = [](std::vector<double> const &v) {
            std::vector<double> o;
            for (double x : v)
                if (x > 0 && x < 1e10) o.push_back(x);
            return o;
        };
        std::printf("case=%s n_query=%d n_ref=%d vpus=%d bucket=%d samples=%zu profiler_samples=%zu "
                    "sum_d2_wrong_runs=%lld\n",
                    argv[2], nq, nr, vpus, Knn1BucketS16::kBucketSize, counters.size(), totNs.size(),
                    static_cast<long long>(badRuns));
        std::printf("total_ms p50 %.3f p99 %.3f | exec_ms p50 %.3f | wall_ms p50 %.3f p99 %.3f\n",
                    median(sane(totNs)) / 1e6, percentile(sane(totNs), 99) / 1e6, median(sane(exeNs)) / 1e6,
                    median(wallNs) / 1e6, percentile(wallNs, 99) / 1e6);
        for (int32_t v = 0; v < vpus; v++)
        {
            std::printf("vpu%d:", v);
            for (auto const &p : kPhases)
            {
                std::vector<double> x;
                for (auto const &c : counters) x.push_back(c[16 * v + p.slot]);
                std::printf(" %s=%.0f", p.name, median(x));
            }
            std::printf("\n");
        }
        return badRuns == 0 ? 0 : 1;
    }
    catch (cupva::Exception const &e)
    {
        std::fprintf(stderr, "cuPVA exception: %s\n", e.what());
        return 1;
    }
}
