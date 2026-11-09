#ifndef GDN_FINALIZE_ENABLE_ARCH22
#define GDN_FINALIZE_ENABLE_ARCH22 0
#endif

#include "aclnn_chunk_gated_delta_rule_bwd_finalize.h"
#include "chunk_gated_delta_rule_bwd_finalize.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#if GDN_FINALIZE_ENABLE_ARCH22
#include "external/aclnn_kernels/aclnn_platform.h"
#endif
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "../../../../../common/host_utils/log.h"

#if GDN_FINALIZE_ENABLE_ARCH22
#include <cstring>

#include "../arch22/aclnn_chunk_gated_delta_rule_bwd_finalize_arch22.h"
#endif

using namespace op;

static constexpr const char *CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ENTITY = "aclnnChunkGatedDeltaRuleBwdFinalize";

namespace {

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr || IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkGatedDeltaRuleBwdFinalizeGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *dO, const aclTensor *du,
    const aclTensor *g, const aclTensor *beta, const aclTensor *h,
    const aclTensor *dh, const aclTensor *a,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclTensor *betaRawOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool useQkL2NormInKernel,
    bool useBetaSigmoidInKernel, bool useGateInKernel, bool stateVFirst,
    bool useExp2,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleBwdFinalize,
        DFX_IN(q, k, v, vNew, dO, du, g, beta, h, dh, a,
               qRstdOptional, kRstdOptional, betaRawOptional,
               cuSeqlensOptional, chunkIndicesOptional),
        DFX_OUT(dqOut, dkOut, dvOut, dbetaOut, dgOut));


    const std::array<const char *, 11> required = {
        "q", "k", "v", "vNew", "dO", "du", "g", "beta", "h", "dh", "a"};
    const std::array<const aclTensor *, 11> requiredTensors = {
        q, k, v, vNew, dO, du, g, beta, h, dh, a};
    for (size_t idx = 0; idx < required.size(); ++idx) {
        if (requiredTensors[idx] == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ENTITY, required[idx]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    const std::array<const char *, 5> gradNames = {"dqOut", "dkOut", "dvOut", "dbetaOut", "dgOut"};
    const std::array<const aclTensor *, 5> gradTensors = {dqOut, dkOut, dvOut, dbetaOut, dgOut};
    for (size_t idx = 0; idx < gradNames.size(); ++idx) {
        if (gradTensors[idx] == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ENTITY, gradNames[idx]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (workspaceSize == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ENTITY, "executor");

        return ACLNN_ERR_PARAM_NULLPTR;
    }
#if GDN_FINALIZE_ENABLE_ARCH22
    // A2/A3 使用私有 Host 路径；A5 继续执行原有校验、连续化和 executor 流程。
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        return gdn_finalize_arch22::GetWorkspaceSize(
            q, k, v, vNew, dO, du, g, beta, h, dh, a,
            qRstdOptional, kRstdOptional, betaRawOptional,
            cuSeqlensOptional, chunkIndicesOptional,
            scale, chunkSize, useQkL2NormInKernel,
            useBetaSigmoidInKernel, useGateInKernel, stateVFirst, useExp2,
            dqOut, dkOut, dvOut, dbetaOut, dgOut, workspaceSize, executor);
    }
    const char *socName = aclrtGetSocName();
    CHECK_COND(socName != nullptr && std::strstr(socName, "Ascend950") != nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "ChunkGatedDeltaRuleBwdFinalize supports Ascend910B, Ascend910C and Ascend950.");
#endif

    CHECK_COND(chunkSize == 64, ACLNN_ERR_PARAM_INVALID, "chunkSize only supports 64.");
    CHECK_COND(!useGateInKernel, ACLNN_ERR_PARAM_INVALID,
               "useGateInKernel only supports false.");
    CHECK_COND((cuSeqlensOptional == nullptr) == (chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional and chunkIndicesOptional must be both present or absent.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();

    CHECK_RET(MakeContiguous(q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(vNew, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(dO, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(du, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(h, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(dh, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(a, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(qRstdOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(kRstdOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(betaRawOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const auto result = l0op::ChunkGatedDeltaRuleBwdFinalize(
        q, k, v, vNew, dO, du, g, beta, h, dh, a,
        qRstdOptional, kRstdOptional, betaRawOptional,
        cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, useQkL2NormInKernel,
        useBetaSigmoidInKernel, useGateInKernel, stateVFirst, useExp2,
        dqOut, dkOut, dvOut, dbetaOut, dgOut,
        executorPtr);
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *outputs[] = {dqOut, dkOut, dvOut, dbetaOut, dgOut};
    for (size_t index = 0; index < result.size(); ++index) {
        const aclTensor *copied = l0op::ViewCopy(result[index], outputs[index], executorPtr);
        CHECK_RET(copied != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleBwdFinalize(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleBwdFinalize);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkGatedDeltaRuleBwdFinalize launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
