#include "aclnn_chunk_kda_bwd_prepare.h"
#include "chunk_kda_bwd_prepare.h"
#include "../../../../common/host_utils/log.h"

#include <cmath>

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    const auto a = lhs->GetViewShape();
    const auto b = rhs->GetViewShape();
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (size_t idx = 0; idx < a.GetDimNum(); ++idx) {
        if (a.GetDim(idx) != b.GetDim(idx)) {
            return false;
        }
    }
    text << ']';
    return text.str();
}

aclnnStatus Missing(const char *name)
{
    // CANN 9.0.0's OP_LOGE_WITH_INVALID_INPUT references an undefined param_name.
    // Use its structured reporter directly without redefining SDK macros.
    OP_LOGE_WITHOUT_REPORT(OP_NAME, "OP[%s] get [%s] failed.", OP_NAME, name);
    const std::vector<const char *> keys = {"op_name", "param_name"};
    const std::vector<const char *> values = {OP_NAME, name};
    REPORT_PREDEFINED_ERR_MSG("EZ0004", keys, values);
    return ACLNN_ERR_PARAM_NULLPTR;
}

aclnnStatus InvalidValue(const char *name, const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_VALUE
    OP_LOGE_FOR_INVALID_VALUE(OP_NAME, name, actual, expected);
#else
    OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(OP_NAME, name, actual, expected);
#endif
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus InvalidArgument(const char *name, const std::string &reason)
{
    return InvalidValue(name, "invalid", reason);
}

aclnnStatus InvalidShape(const aclTensor *tensor, const char *name, const std::string &expected)
{
    const auto actual = ShapeText(tensor->GetViewShape());
#ifdef OP_LOGE_FOR_INVALID_SHAPE
    // The predefined message supplies the enclosing brackets itself.
    const auto shape = actual.substr(1, actual.size() - 2);
    const auto correct = expected.size() >= 2 && expected.front() == '[' && expected.back() == ']'
        ? expected.substr(1, expected.size() - 2) : expected;
    OP_LOGE_FOR_INVALID_SHAPE(OP_NAME, name, shape, correct);
#else
    // Older SDKs have no EZ0008 catalog entry. Keep details in their EZ0002 report.
    OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(OP_NAME, name, actual, expected);
#endif
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus CheckTensor(const aclTensor *tensor, DataType dtype, const char *name)
{
    if (tensor == nullptr) return Missing(name);
    if (tensor->GetDataType() != dtype) {
#ifdef OP_LOGE_FOR_INVALID_DTYPE
        OP_LOGE_FOR_INVALID_DTYPE(OP_NAME, name,
            ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()),
            ge::TypeUtils::DataTypeToSerialString(dtype));
#else
        const auto actual = ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType());
        const auto expected = ge::TypeUtils::DataTypeToSerialString(dtype);
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(OP_NAME, name, actual, expected);
#endif
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsContiguous(tensor)) return InvalidArgument(name, "must be contiguous");
    return ACLNN_SUCCESS;
}
aclnnStatus ValidateMetadata(
    const aclIntArray *cu, const aclIntArray *indices, int64_t totalTokens,
    int64_t chunkSize, int64_t expectedChunks)
{
    CHECK_COND(cu != nullptr && indices != nullptr, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must be provided together");
    CHECK_COND(cu->Size() >= 2 && cu->Size() <= 1025, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must contain 2..1025 values");
    CHECK_COND((*cu)[0] == 0 && (*cu)[cu->Size() - 1] == totalTokens,
               ACLNN_ERR_PARAM_INVALID, "cu_seqlens must start at zero and end at T");
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
        const int64_t begin = (*cu)[seq];
        const int64_t end = (*cu)[seq + 1];
        CHECK_COND(begin >= 0 && end >= begin, ACLNN_ERR_PARAM_INVALID,
                   "cu_seqlens must be nondecreasing");
        totalChunks += (end - begin + chunkSize - 1) / chunkSize;
    }
    CHECK_COND(totalChunks == expectedChunks && totalChunks > 0, ACLNN_ERR_PARAM_INVALID,
               "h chunk dimension does not match cu_seqlens");
    CHECK_COND(indices->Size() == static_cast<size_t>(2 * totalChunks), ACLNN_ERR_PARAM_INVALID,
               "chunk_indices must contain exactly one pair per chunk");
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
        const int64_t count = ((*cu)[seq + 1] - (*cu)[seq] + chunkSize - 1) / chunkSize;
        for (int64_t local = 0; local < count; ++local) {
            CHECK_COND((*indices)[offset] == static_cast<int64_t>(seq) &&
                           (*indices)[offset + 1] == local,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunk_indices must use canonical sequence-major order");
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Check(
    const aclTensor *aqk, const aclTensor *vNew, const aclTensor *dO, const aclTensor *h,
    const aclIntArray *cu, const aclIntArray *indices, double scale, int64_t chunkSize,
    const aclTensor *dAqkOut, const aclTensor *dvOut, const aclTensor *dqRawOut)
{
    CHECK_RET(CheckTensor(aqk, DataType::DT_BF16, "aqk") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensor(vNew, DataType::DT_BF16, "v_new") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensor(dO, DataType::DT_BF16, "d_o") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensor(h, DataType::DT_BF16, "h") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensor(dAqkOut, DataType::DT_FLOAT, "d_aqk") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensor(dvOut, DataType::DT_BF16, "dv") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensor(dqRawOut, DataType::DT_FLOAT, "dq_raw") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(std::isfinite(scale), ACLNN_ERR_PARAM_INVALID, "scale must be finite");
    CHECK_COND(chunkSize == 64, ACLNN_ERR_PARAM_INVALID, "chunk_size must be 64");
    CHECK_COND((cu == nullptr) == (indices == nullptr), ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must be provided together");

    const bool variable = cu != nullptr;
    const auto a = aqk->GetViewShape();
    const auto v = vNew->GetViewShape();
    const auto o = dO->GetViewShape();
    const auto state = h->GetViewShape();
    const size_t rank = variable ? 3 : 4;
    CHECK_COND(a.GetDimNum() == rank && v.GetDimNum() == rank && o.GetDimNum() == rank,
               ACLNN_ERR_PARAM_INVALID, "token tensor rank is invalid");
    CHECK_COND(state.GetDimNum() == rank + 1, ACLNN_ERR_PARAM_INVALID, "h rank is invalid");

    const int64_t B = variable ? 1 : a.GetDim(0);
    const int64_t NV = variable ? a.GetDim(0) : a.GetDim(1);
    const int64_t T = variable ? a.GetDim(1) : a.GetDim(2);
    const int64_t C = variable ? a.GetDim(2) : a.GetDim(3);
    const int64_t vHead = variable ? v.GetDim(0) : v.GetDim(1);
    const int64_t vT = variable ? v.GetDim(1) : v.GetDim(2);
    const int64_t V = variable ? v.GetDim(2) : v.GetDim(3);
    const int64_t oHead = variable ? o.GetDim(0) : o.GetDim(1);
    const int64_t oT = variable ? o.GetDim(1) : o.GetDim(2);
    const int64_t oV = variable ? o.GetDim(2) : o.GetDim(3);
    CHECK_COND(B > 0 && NV > 0 && T > 0 && C == 64 && V == 128 && oV == 128,
               ACLNN_ERR_PARAM_INVALID, "KernelA supports C=64 and K=V=128 only");
    CHECK_COND(vHead == NV && oHead == NV && vT == T && oT == T,
               ACLNN_ERR_PARAM_INVALID, "aqk/v_new/d_o head or token dimensions mismatch");

    const int64_t chunkCount = variable ? state.GetDim(0) : state.GetDim(1);
    const bool stateShapeValid = variable
        ? state.GetDim(1) == NV && state.GetDim(2) == 128 && state.GetDim(3) == 128
        : state.GetDim(0) == B && state.GetDim(2) == NV &&
              state.GetDim(3) == 128 && state.GetDim(4) == 128;
    CHECK_COND(stateShapeValid, ACLNN_ERR_PARAM_INVALID, "h shape is invalid");
    if (variable) {
        CHECK_RET(ValidateMetadata(cu, indices, T, chunkSize, chunkCount) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_COND(chunkCount == (T + chunkSize - 1) / chunkSize,
                   ACLNN_ERR_PARAM_INVALID, "dense h chunk dimension is invalid");
    }
    CHECK_COND(SameShape(aqk, dAqkOut), ACLNN_ERR_PARAM_INVALID,
               "d_aqk output shape must match aqk");
    CHECK_COND(SameShape(vNew, dvOut) && SameShape(dO, dqRawOut), ACLNN_ERR_PARAM_INVALID,
               "dv/dq_raw output shape mismatch");
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdPrepareGetWorkspaceSize(
    const aclTensor *aqk, const aclTensor *vNew, const aclTensor *dO, const aclTensor *h,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool stateVFirst,
    const aclTensor *dAqkOut, const aclTensor *dvOut, const aclTensor *dqRawOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const char *inputLayout = layout;

    // DFX also checks these pointers, but its generic error omits the operator
    // and parameter names. Report them before entering the DFX helper.
    if (workspaceSize == nullptr) return Missing("workspaceSize");
    if (executor == nullptr) return Missing("executor");
    L2_DFX_PHASE_1(aclnnChunkKdaBwdPrepare,
                   DFX_IN(aqk, vNew, dO, h, cuSeqlensOptional, chunkIndicesOptional,
                          scale, chunkSize, stateVFirst),
                   DFX_OUT(dAqkOut, dvOut, dqRawOut));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "workspaceSize and executor must not be nullptr");
    CHECK_RET(Check(aqk, vNew, dO, h, cuSeqlensOptional, chunkIndicesOptional,
                    scale, chunkSize, dAqkOut, dvOut, dqRawOut) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto result = l0op::ChunkKdaBwdPrepare(
        aqk, vNew, dO, h, cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, stateVFirst, dAqkOut, dvOut, dqRawOut,
        uniqueExecutor.get());
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdPrepare(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdPrepare);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkKdaBwdPrepare launch failed");
    return ACLNN_SUCCESS;
}
