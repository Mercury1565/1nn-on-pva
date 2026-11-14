/* Performance harness, hardware only (PERF-KITTI-A/B). One CSV row per sample:
 * prepare, submit->fence wall, PVA profiler total/exec/overhead, results, e2e (ns).
 * usage: knn1_bucket_s16_perf <testdata_dir> <out_csv> [--case kitti_a] [--warmup 100] [--samples 1000]
 *                             [--vpus 2] [--threads 1] */
#include "knn1_bucket_s16.hpp"
#include "knn1_ref.hpp"

#include <cupva_host.hpp>
#include <cupva_host_utils.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using pva_knn::Knn1BucketS16;
using Clock = std::chrono::steady_clock;

static std::vector<char> readAll(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static double ns(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::nano>(b - a).count();
}

static double pct(std::vector<double> v, double p)
{
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    const double pos = p / 100.0 * (v.size() - 1);
    const size_t lo  = static_cast<size_t>(pos);
    const size_t hi  = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

static std::vector<double> profilerNs(cupva_utils::ProfilingContext &ctx, cupva_utils::ProfilingBatchStatType t)
{
    auto const &d = ctx.getStatistics(t).getStatsData();
    std::vector<double> v;
    for (int64_t i = 0; d.data != nullptr && i < static_cast<int64_t>(d.count); i++) v.push_back(double(d.data[i]));
    return v;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <testdata_dir> <out_csv> [--case kitti_a] [--warmup N] [--samples N] "
                             "[--vpus N] [--threads N]\n",
                     argv[0]);
        return 2;
    }
    const std::string root = argv[1], out = argv[2];
    std::string cs = "kitti_a";
    int warmup = 100, samples = 1000, vpus = 2, threads = 1;
    for (int a = 3; a + 1 < argc; a += 2)
    {
        const std::string k = argv[a];
        if (k == "--case") cs = argv[a + 1];
        else if (k == "--warmup") warmup = std::atoi(argv[a + 1]);
        else if (k == "--samples") samples = std::atoi(argv[a + 1]);
        else if (k == "--vpus") vpus = std::atoi(argv[a + 1]);
        else if (k == "--threads") threads = std::atoi(argv[a + 1]);
    }

    /* Load the frame pair and its expected distances. */
    const std::string dir = root + "/" + cs;
    std::vector<char> qr = readAll(dir + "/queries.s16"), rr = readAll(dir + "/refs.s16"),
                      er = readAll(dir + "/dist2.s32");
    if (qr.empty() || rr.empty() || er.size() != qr.size() / 6 * 4)
    {
        std::fprintf(stderr, "cannot read %s\n", dir.c_str());
        return 2;
    }
    const int32_t nq = static_cast<int32_t>(qr.size() / 6), nr = static_cast<int32_t>(rr.size() / 6);
    std::vector<int16_t> q(3 * static_cast<size_t>(nq)), r(3 * static_cast<size_t>(nr));
    std::vector<int32_t> exp(static_cast<size_t>(nq)), d2(static_cast<size_t>(nq)), idx(static_cast<size_t>(nq));
    std::memcpy(q.data(), qr.data(), qr.size());
    std::memcpy(r.data(), rr.data(), rr.size());
    std::memcpy(exp.data(), er.data(), er.size());

    try
    {
        Knn1BucketS16 op;
        if (op.init(vpus, threads) != 0)
        {
            std::fprintf(stderr, "init failed\n");
            return 1;
        }
        cupva_utils::ProfilingContext ctx = cupva_utils::ProfilingContext::Create();
        cupva::Stream stream              = cupva_utils::CreateProfilingStream(ctx);
        cupva::SyncObj sync               = cupva::SyncObj::Create();

        struct Sample
        {
            double prepare, wall, results;
        };
        auto once = [&](Sample *s) -> bool {
            const auto t0 = Clock::now();
            if (op.prepare(q.data(), nq, r.data(), nr) != 0) return false;
            const auto t1 = Clock::now();
            cupva::Fence fence{sync};
            if (op.submit(stream, fence) != 0) return false;
            fence.wait();
            const auto t2 = Clock::now();
            if (op.lastStatus() != cupva::Error::None || op.results(d2.data(), idx.data()) != 0) return false;
            const auto t3 = Clock::now();
            if (s) *s = Sample{ns(t0, t1), ns(t1, t2), ns(t2, t3)};
            return true;
        };

        for (int i = 0; i < warmup; i++)
        {
            if (!once(nullptr))
            {
                std::fprintf(stderr, "warm-up call %d failed\n", i);
                return 1;
            }
        }
        ctx.reset();

        std::vector<Sample> s(static_cast<size_t>(samples));
        for (int i = 0; i < samples; i++)
        {
            if (!once(&s[i]))
            {
                std::fprintf(stderr, "timed call %d failed\n", i);
                return 1;
            }
        }
        const auto tot = profilerNs(ctx, cupva_utils::ProfilingBatchStatType::TOTAL_TIME);
        const auto exe = profilerNs(ctx, cupva_utils::ProfilingBatchStatType::EXECUTION_TIME);
        const auto ovh = profilerNs(ctx, cupva_utils::ProfilingBatchStatType::TASK_OVERHEAD);
        pva_knn::Knn1Stats st{};
        op.stats(&st);

        /* Correctness gate on the last timed call. */
        int64_t bad = 0;
        for (int32_t i = 0; i < nq; i++)
        {
            bad += (d2[i] != exp[i] || idx[i] < 0 || idx[i] >= nr ||
                    pva_knn_ref::dist2(q.data(), nq, i, r.data(), nr, idx[i]) != d2[i]);
        }

        std::ofstream f(out);
        f << "sample,prepare_ns,wall_ns,total_ns,exec_ns,overhead_ns,results_ns,e2e_ns\n";
        auto at = [](std::vector<double> const &v, int i) { return i < static_cast<int>(v.size()) ? v[i] : -1.0; };
        for (int i = 0; i < samples; i++)
        {
            f << i << "," << s[i].prepare << "," << s[i].wall << "," << at(tot, i) << "," << at(exe, i) << ","
              << at(ovh, i) << "," << s[i].results << "," << (s[i].prepare + s[i].wall + s[i].results) << "\n";
        }

        auto us = [](std::vector<double> const &v) {
            std::vector<double> o;
            for (double x : v)
                if (x > 0 && x < 1e10) o.push_back(x / 1000.0); /* valid profiler entries only */
            return o;
        };
        std::vector<double> prep, wall, res, e2e;
        for (auto const &x : s)
        {
            prep.push_back(x.prepare / 1000.0);
            wall.push_back(x.wall / 1000.0);
            res.push_back(x.results / 1000.0);
            e2e.push_back((x.prepare + x.wall + x.results) / 1000.0);
        }
        auto line = [](const char *name, std::vector<double> const &v) {
            std::printf("%-12s n=%zu p50_us=%.1f p99_us=%.1f min_us=%.1f max_us=%.1f\n", name, v.size(), pct(v, 50),
                        pct(v, 99), pct(v, 0), pct(v, 100));
        };
        line("prepare", prep);
        line("pva_wall", wall);
        line("pva_total", us(tot));
        line("vpu_exec_sum", us(exe));
        line("overhead", us(ovh));
        line("results", res);
        line("e2e", e2e);
        std::printf("case=%s n_query=%d n_ref=%d bucket=%d vpus=%d threads=%d teams=%d buckets=%d "
                    "buckets_fetched=%lld dma_fetches=%lld warmup=%d samples=%d profiler_samples=%zu (valid %zu) "
                    "correctness_gate=%s (bad=%lld)\n",
                    cs.c_str(), nq, nr, Knn1BucketS16::kBucketSize, vpus, threads, st.n_teams, st.n_buckets,
                    static_cast<long long>(st.buckets_fetched), static_cast<long long>(st.dma_fetches), warmup,
                    samples, tot.size(), us(tot).size(), bad == 0 ? "PASS" : "FAIL", static_cast<long long>(bad));
        return bad == 0 ? 0 : 1;
    }
    catch (cupva::Exception const &e)
    {
        std::fprintf(stderr, "cuPVA exception: %s\n", e.what());
        return 1;
    }
}
