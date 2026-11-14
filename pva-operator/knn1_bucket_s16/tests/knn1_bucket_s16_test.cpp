/* Correctness tests (configure/test-matrix.json): exact dist2, valid idx, sum, guards, inputs unchanged.
 * usage: knn1_bucket_s16_test <testdata_dir> [--case ID]... [--list] */
#include "knn1_bucket_s16.hpp"
#include "knn1_ref.hpp"

#include <cupva_host.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using pva_knn::Knn1BucketS16;
using pva_knn::Knn1Stats;

namespace {

constexpr int32_t kSentinel = 0x7FFFFFFF;
constexpr int32_t kGuard    = 64; /* extra output elements that must stay untouched */

const char *const kDataCases[] = {
    "kitti_a",         "kitti_b",          "kitti_c",          "kitti_d",           "kitti_a_swap",
    "kitti_16k",       "kitti_unequal",    "kitti_self",       "max_131072",        "small_1x1",
    "small_1x512",     "small_16x1",       "small_15x511",     "small_16x512",      "small_17x513",
    "small_513x17",    "small_1000x3000",  "extremes",         "duplicates_grid",   "all_refs_identical",
    "degenerate_line", "degenerate_plane", "far_clusters"};

const char *const kOtherCases[] = {"CORR-REPEATED-SUBMISSION", "CORR-REPREPARE-SIZES", "CORR-ASYNC-FENCE",
                                   "CORR-SUMDIST2-INT64",      "CORR-STATS-VS-MODEL",  "ERR-ORDER",
                                   "ERR-ARGS",                 "CORR-NO-ALIAS"};

int g_pass = 0, g_fail = 0;

void report(const std::string &id, bool ok, const std::string &detail)
{
    std::printf("RESULT %s %s %s\n", id.c_str(), ok ? "PASS" : "FAIL", detail.c_str());
    std::fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

std::string caseId(const std::string &name)
{
    std::string id = "CORR-" + name;
    for (char &c : id)
    {
        c = (c == '_') ? '-' : static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return id;
}

/* ---------------------------------------------------------------- test data */

struct Case
{
    std::string name;
    int32_t nq{0}, nr{0};
    std::vector<int16_t> q, r;  /* planar [3][n] */
    std::vector<int32_t> expD2; /* [nq] */
    int64_t manifestSum{-1};
};

std::vector<char> readAll(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

/* Minimal manifest lookup: the "sum_dist2" that follows "case": "<name>". */
int64_t manifestSum(const std::string &dir, const std::string &name)
{
    std::vector<char> raw = readAll(dir + "/manifest.json");
    const std::string m(raw.begin(), raw.end());
    const size_t at = m.find("\"case\": \"" + name + "\"");
    if (at == std::string::npos) return -1;
    const size_t s = m.find("\"sum_dist2\": ", at);
    if (s == std::string::npos) return -1;
    return std::stoll(m.substr(s + 13, 24));
}

bool loadCase(const std::string &dir, const std::string &name, Case &c)
{
    const std::string d = dir + "/" + name;
    std::vector<char> q = readAll(d + "/queries.s16"), r = readAll(d + "/refs.s16"), e = readAll(d + "/dist2.s32");
    if (q.empty() || r.empty() || q.size() % 6 || r.size() % 6 || e.size() != q.size() / 6 * 4) return false;
    c.name = name;
    c.nq   = static_cast<int32_t>(q.size() / 6);
    c.nr   = static_cast<int32_t>(r.size() / 6);
    c.q.resize(3 * static_cast<size_t>(c.nq));
    c.r.resize(3 * static_cast<size_t>(c.nr));
    c.expD2.resize(static_cast<size_t>(c.nq));
    std::memcpy(c.q.data(), q.data(), q.size());
    std::memcpy(c.r.data(), r.data(), r.size());
    std::memcpy(c.expD2.data(), e.data(), e.size());
    c.manifestSum = manifestSum(dir, name);
    return c.manifestSum >= 0;
}

/* ---------------------------------------------------------------- one run and its check */

struct Run
{
    int32_t rc{0};
    cupva::Error status{cupva::Error::None};
    std::vector<int32_t> d2, idx; /* [nq + kGuard], sentinel-prefilled */
    Knn1Stats st{};
};

Run runOnce(Knn1BucketS16 &op, cupva::Stream &stream, cupva::SyncObj &sync, Case const &c)
{
    Run out;
    out.d2.assign(static_cast<size_t>(c.nq) + kGuard, kSentinel);
    out.idx.assign(static_cast<size_t>(c.nq) + kGuard, kSentinel);
    if ((out.rc = op.prepare(c.q.data(), c.nq, c.r.data(), c.nr)) != 0) return out;
    cupva::Fence fence{sync};
    if ((out.rc = op.submit(stream, fence)) != 0) return out;
    fence.wait();
    out.status = op.lastStatus();
    if (out.status != cupva::Error::None) return out;
    if ((out.rc = op.results(out.d2.data(), out.idx.data())) != 0) return out;
    out.rc = op.stats(&out.st);
    return out;
}

/* Returns "" if the run is correct, else a description of the first problem. */
std::string checkRun(Case const &c, Run const &run)
{
    std::ostringstream why;
    if (run.rc != 0) return "api returned " + std::to_string(run.rc);
    if (run.status != cupva::Error::None) return "cupva status " + std::to_string(static_cast<int>(run.status));
    int64_t badD2 = 0, badIdx = 0;
    for (int32_t i = 0; i < c.nq; i++)
    {
        badD2 += (run.d2[i] != c.expD2[i]);
        const int32_t j = run.idx[i];
        badIdx += (j < 0 || j >= c.nr || pva_knn_ref::dist2(c.q.data(), c.nq, i, c.r.data(), c.nr, j) != run.d2[i]);
    }
    int64_t guardTouched = 0;
    for (int32_t g = 0; g < kGuard; g++)
    {
        guardTouched += (run.d2[c.nq + g] != kSentinel) + (run.idx[c.nq + g] != kSentinel);
    }
    const int64_t sum   = Knn1BucketS16::sumDist2(run.d2.data(), c.nq);
    const int64_t pad   = ((run.st.n_buckets + 15) / 16) * 16;
    const bool statsOk  = run.st.n_teams == (c.nq + 15) / 16 && run.st.buckets_fetched >= run.st.n_teams &&
                         run.st.buckets_fetched <= static_cast<int64_t>(run.st.n_teams) * pad;
    if (badD2) why << "dist2_mismatch=" << badD2 << " ";
    if (badIdx) why << "idx_invalid=" << badIdx << " ";
    if (guardTouched) why << "wrote_past_n_query=" << guardTouched << " ";
    if (sum != c.manifestSum) why << "sum_dist2=" << sum << "!=" << c.manifestSum << " ";
    if (!statsOk) why << "stats_out_of_bounds fetched=" << run.st.buckets_fetched << " ";
    return why.str();
}

std::string runSummary(Case const &c, Run const &run)
{
    std::ostringstream s;
    s << "nq=" << c.nq << " nr=" << c.nr << " teams=" << run.st.n_teams << " buckets=" << run.st.n_buckets
      << " fetched=" << run.st.buckets_fetched << " fetched_per_team="
      << (run.st.n_teams ? double(run.st.buckets_fetched) / run.st.n_teams : 0.0) << " sum_dist2="
      << Knn1BucketS16::sumDist2(run.d2.data(), c.nq);
    return s.str();
}

/* Small cases: the expected file must agree with the independent C++ brute force too. */
std::string crossCheckExpected(Case const &c)
{
    if (static_cast<int64_t>(c.nq) * c.nr > 50000000LL) return "";
    std::vector<int64_t> d(static_cast<size_t>(c.nq));
    std::vector<int32_t> j(static_cast<size_t>(c.nq));
    pva_knn_ref::nearest(c.q.data(), c.nq, c.r.data(), c.nr, d.data(), j.data());
    for (int32_t i = 0; i < c.nq; i++)
    {
        if (d[i] != c.expD2[i]) return "expected_file_disagrees_with_cpp_reference ";
    }
    return "";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <testdata_dir> [--case ID]... [--list]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    std::set<std::string> only;
    for (int a = 2; a < argc; a++)
    {
        const std::string k = argv[a];
        if (k == "--list")
        {
            for (const char *n : kDataCases) std::printf("%s\n", caseId(n).c_str());
            for (const char *n : kOtherCases) std::printf("%s\n", n);
            return 0;
        }
        if (k == "--case" && a + 1 < argc) only.insert(argv[++a]);
    }
    auto selected = [&](const std::string &id) { return only.empty() || only.count(id) > 0; };

    std::map<std::string, Case> cache;
    auto get = [&](const std::string &name) -> Case const & {
        auto it = cache.find(name);
        if (it == cache.end())
        {
            Case c;
            if (!loadCase(dir, name, c)) throw std::runtime_error("cannot load case " + name);
            it = cache.emplace(name, std::move(c)).first;
        }
        return it->second;
    };

    try
    {
        cupva::Stream stream = cupva::Stream::Create();
        cupva::SyncObj sync  = cupva::SyncObj::Create();
        Knn1BucketS16 op;
        if (op.init() != 0)
        {
            std::printf("RESULT HARNESS FAIL init failed\n");
            return 1;
        }
        std::map<std::string, Run> results; /* data-case runs, reused by later checks */

        /* ---- data cases (one shared instance: also exercises re-prepare between sizes) */
        for (const char *n : kDataCases)
        {
            if (!selected(caseId(n))) continue;
            Case const &c = get(n);
            Run run       = runOnce(op, stream, sync, c);
            std::string why = checkRun(c, run) + crossCheckExpected(c);
            report(caseId(n), why.empty(), why.empty() ? runSummary(c, run) : why);
            results[n] = std::move(run);
        }

        /* ---- CORR-REPEATED-SUBMISSION: 100 submissions of kitti_a, bit-identical dist2 */
        if (selected("CORR-REPEATED-SUBMISSION"))
        {
            Case const &c = get("kitti_a");
            std::string why;
            if (op.prepare(c.q.data(), c.nq, c.r.data(), c.nr) != 0) why = "prepare failed";
            std::vector<int32_t> first;
            for (int k = 0; k < 100 && why.empty(); k++)
            {
                cupva::Fence fence{sync};
                if (op.submit(stream, fence) != 0) { why = "submit failed"; break; }
                fence.wait();
                Run run;
                run.d2.assign(static_cast<size_t>(c.nq) + kGuard, kSentinel);
                run.idx.assign(static_cast<size_t>(c.nq) + kGuard, kSentinel);
                run.status = op.lastStatus();
                run.rc     = (run.status == cupva::Error::None) ? op.results(run.d2.data(), run.idx.data()) : -1;
                if (run.rc == 0) run.rc = op.stats(&run.st);
                why = checkRun(c, run);
                if (!why.empty()) { why = "submission " + std::to_string(k) + ": " + why; break; }
                if (k == 0) first = run.d2;
                else if (run.d2 != first) why = "submission " + std::to_string(k) + " differs from the first";
            }
            report("CORR-REPEATED-SUBMISSION", why.empty(), why.empty() ? "100 submissions identical and correct" : why);
        }

        /* ---- CORR-REPREPARE-SIZES: large -> small -> large on one instance, no stale data */
        if (selected("CORR-REPREPARE-SIZES"))
        {
            std::string why;
            for (const char *n : {"max_131072", "small_17x513", "kitti_b", "small_1x1", "max_131072"})
            {
                Case const &c = get(n);
                why           = checkRun(c, runOnce(op, stream, sync, c));
                if (!why.empty()) { why = std::string(n) + ": " + why; break; }
            }
            report("CORR-REPREPARE-SIZES", why.empty(), why.empty() ? "5-step sequence correct" : why);
        }

        /* ---- CORR-ASYNC-FENCE: results read only after the fence, into sentinel-filled buffers */
        if (selected("CORR-ASYNC-FENCE"))
        {
            Case const &c = get("kitti_16k");
            Run run       = runOnce(op, stream, sync, c);
            std::string why = checkRun(c, run);
            int64_t sentinels = 0;
            for (int32_t i = 0; i < c.nq; i++) sentinels += (run.d2[i] == kSentinel) + (run.idx[i] == kSentinel);
            if (sentinels) why += "sentinel_left=" + std::to_string(sentinels);
            report("CORR-ASYNC-FENCE", why.empty(), why.empty() ? "no sentinel left, case correct" : why);
        }

        /* ---- CORR-SUMDIST2-INT64: sums above 2^31 are exact */
        if (selected("CORR-SUMDIST2-INT64"))
        {
            std::string why;
            for (const char *n : {"extremes", "all_refs_identical", "max_131072"})
            {
                Case const &c = get(n);
                if (!results.count(n)) results[n] = runOnce(op, stream, sync, c);
                const int64_t s = Knn1BucketS16::sumDist2(results[n].d2.data(), c.nq);
                if (s != c.manifestSum || s <= INT32_MAX)
                    why += std::string(n) + ":" + std::to_string(s) + "!=" + std::to_string(c.manifestSum) + " ";
            }
            report("CORR-SUMDIST2-INT64", why.empty(), why.empty() ? "3 sums > 2^31 exact" : why);
        }

        /* ---- CORR-STATS-VS-MODEL: informational fetch counts (compare offline with the Python model) */
        if (selected("CORR-STATS-VS-MODEL"))
        {
            std::ostringstream s;
            bool ok = true;
            for (const char *n : {"kitti_a", "kitti_b", "kitti_c", "kitti_d"})
            {
                Case const &c = get(n);
                if (!results.count(n)) results[n] = runOnce(op, stream, sync, c);
                Run const &r = results[n];
                const int64_t pad = ((r.st.n_buckets + 15) / 16) * 16;
                ok &= (r.rc == 0 && r.st.buckets_fetched >= r.st.n_teams &&
                       r.st.buckets_fetched <= static_cast<int64_t>(r.st.n_teams) * pad);
                s << n << ":fetched=" << r.st.buckets_fetched << ",teams=" << r.st.n_teams << " ";
            }
            report("CORR-STATS-VS-MODEL", ok, s.str());
        }

        /* ---- ERR-ORDER: calls out of order fail and leave caller buffers untouched */
        if (selected("ERR-ORDER"))
        {
            Case const &c = get("small_16x512");
            Knn1BucketS16 fresh;
            std::vector<int32_t> d(16, kSentinel), j(16, kSentinel);
            Knn1Stats st{};
            cupva::Fence fence{sync};
            bool ok = fresh.prepare(c.q.data(), c.nq, c.r.data(), c.nr) != 0; /* before init */
            ok &= fresh.init() == 0;
            ok &= fresh.submit(stream, fence) != 0;      /* before prepare */
            ok &= fresh.results(d.data(), j.data()) != 0; /* before submit */
            ok &= fresh.stats(&st) != 0;
            ok &= std::all_of(d.begin(), d.end(), [](int32_t v) { return v == kSentinel; });
            ok &= std::all_of(j.begin(), j.end(), [](int32_t v) { return v == kSentinel; });
            report("ERR-ORDER", ok, ok ? "all out-of-order calls rejected" : "an out-of-order call was accepted");
        }

        /* ---- ERR-ARGS: bad pointers and sizes fail; the instance still works afterwards */
        if (selected("ERR-ARGS"))
        {
            Case const &c = get("small_16x512");
            bool ok       = true;
            for (int32_t n : {0, -1, Knn1BucketS16::kMaxPoints + 1})
            {
                ok &= op.prepare(c.q.data(), n, c.r.data(), c.nr) != 0;
                ok &= op.prepare(c.q.data(), c.nq, c.r.data(), n) != 0;
            }
            ok &= op.prepare(nullptr, c.nq, c.r.data(), c.nr) != 0;
            ok &= op.prepare(c.q.data(), c.nq, nullptr, c.nr) != 0;
            Run good = runOnce(op, stream, sync, c);
            ok &= checkRun(c, good).empty();
            std::vector<int32_t> d(16);
            ok &= op.results(nullptr, d.data()) != 0;
            ok &= op.results(d.data(), nullptr) != 0;
            ok &= op.stats(nullptr) != 0;
            report("ERR-ARGS", ok, ok ? "bad arguments rejected, later valid run correct" : "a bad argument was accepted");
        }

        /* ---- CORR-NO-ALIAS: separate buffers, inputs unchanged after a full run */
        if (selected("CORR-NO-ALIAS"))
        {
            Case const &c = get("kitti_16k");
            const std::vector<int16_t> q0 = c.q, r0 = c.r;
            Run run = runOnce(op, stream, sync, c);
            const bool disjoint = run.d2.data() != run.idx.data() &&
                                  (run.d2.data() + run.d2.size() <= run.idx.data() ||
                                   run.idx.data() + run.idx.size() <= run.d2.data());
            std::string why = checkRun(c, run);
            if (!disjoint) why += "buffers_overlap ";
            if (c.q != q0 || c.r != r0) why += "inputs_modified ";
            report("CORR-NO-ALIAS", why.empty(), why.empty() ? "disjoint buffers, inputs unchanged" : why);
        }
    }
    catch (cupva::Exception const &e)
    {
        std::printf("RESULT HARNESS FAIL cuPVA exception: %s\n", e.what());
        g_fail++;
    }
    catch (std::exception const &e)
    {
        std::printf("RESULT HARNESS FAIL %s\n", e.what());
        g_fail++;
    }
    std::printf("SUMMARY pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
