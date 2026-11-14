/* C interface over Knn1BucketS16 (see knn1_distance.h). */
#include "knn1_distance.h"

#include "knn1_bucket_s16.hpp"

#include <cupva_host.hpp>

#include <cmath>
#include <new>

#define KNN1_API extern "C" __attribute__((visibility("default")))

namespace {

/* Error codes 1-5 are those of Knn1BucketS16; the rest are added here. */
constexpr int kErrPvaRun   = 6; /* the PVA reported an error for the submission */
constexpr int kErrNullCtx  = 7;
constexpr int kErrInternal = 8;

} // namespace

struct knn1_ctx
{
    pva_knn::Knn1BucketS16 op;
    cupva::Stream stream;
    cupva::SyncObj sync;
    int64_t lastFetched{-1};
};

KNN1_API knn1_ctx *knn1_create(void)
{
    try
    {
        knn1_ctx *ctx = new knn1_ctx;
        if (ctx->op.init() != 0)
        {
            delete ctx;
            return nullptr;
        }
        ctx->stream = cupva::Stream::Create();
        ctx->sync   = cupva::SyncObj::Create();
        return ctx;
    }
    catch (...)
    {
        return nullptr;
    }
}

KNN1_API void knn1_destroy(knn1_ctx *ctx)
{
    delete ctx;
}

KNN1_API int knn1_nearest(knn1_ctx *ctx, const int16_t *pc_queries, int32_t n_query, const int16_t *pc_ref,
                          int32_t n_ref, int32_t *dist2, int32_t *idx, int64_t *sum_d2)
{
    if (ctx == nullptr)
    {
        return kErrNullCtx;
    }
    ctx->lastFetched = -1;
    try
    {
        int rc = ctx->op.prepare(pc_queries, n_query, pc_ref, n_ref);
        if (rc != 0)
        {
            return rc;
        }
        cupva::Fence fence{ctx->sync};
        rc = ctx->op.submit(ctx->stream, fence);
        if (rc != 0)
        {
            return rc;
        }
        fence.wait();
        if (ctx->op.lastStatus() != cupva::Error::None)
        {
            return kErrPvaRun;
        }

        /* idx is optional for the caller, but results() always writes it. */
        int32_t *idxOut = idx;
        int32_t *scratch = nullptr;
        if (idxOut == nullptr)
        {
            scratch = new (std::nothrow) int32_t[n_query];
            if (scratch == nullptr)
            {
                return kErrInternal;
            }
            idxOut = scratch;
        }
        rc = ctx->op.results(dist2, idxOut);
        delete[] scratch;
        if (rc != 0)
        {
            return rc;
        }

        pva_knn::Knn1Stats st{};
        if (ctx->op.stats(&st) == 0)
        {
            ctx->lastFetched = st.buckets_fetched;
        }
        if (sum_d2 != nullptr)
        {
            *sum_d2 = pva_knn::Knn1BucketS16::sumDist2(dist2, n_query);
        }
        return 0;
    }
    catch (...)
    {
        return kErrInternal;
    }
}

KNN1_API int32_t knn1_quantize_cm(const float *xyz, int32_t n, int32_t stride, int16_t *planar_out)
{
    if (xyz == nullptr || planar_out == nullptr || n < 1 || stride < 3)
    {
        return -1;
    }
    int32_t clamped = 0;
    for (int32_t i = 0; i < n; i++)
    {
        for (int32_t a = 0; a < 3; a++)
        {
            float v = xyz[static_cast<int64_t>(i) * stride + a];
            if (v < -120.0f || v > 120.0f)
            {
                v = (v < 0.0f) ? -120.0f : 120.0f;
                clamped++;
            }
            /* Default rounding mode: round half to even (same as numpy.rint). */
            planar_out[static_cast<int64_t>(a) * n + i] =
                static_cast<int16_t>(std::nearbyint(static_cast<double>(v) * 100.0));
        }
    }
    return clamped;
}

KNN1_API int64_t knn1_last_buckets_fetched(const knn1_ctx *ctx)
{
    return ctx == nullptr ? -1 : ctx->lastFetched;
}

KNN1_API const char *knn1_error_string(int code)
{
    switch (code)
    {
    case 0: return "ok";
    case 1: return "call out of order";
    case 2: return "bad argument (null pointer, or a cloud size outside 1..131072)";
    case 3: return "cuPVA error";
    case 4: return "unexpected error";
    case 5: return "PVA output failed its consistency check";
    case kErrPvaRun: return "the PVA reported an error for this run";
    case kErrNullCtx: return "context is NULL";
    case kErrInternal: return "internal error (out of memory or exception)";
    default: return "unknown error code";
    }
}
