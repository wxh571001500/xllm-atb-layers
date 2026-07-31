/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "mla_fused_infer_attention_operation.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v4.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {
namespace {

constexpr uint32_t K_REQUIRED_INPUT_NUM = 7;
constexpr uint32_t K_QUERY_INDEX = 0;
constexpr uint32_t K_KEY_INDEX = 1;
constexpr uint32_t K_VALUE_INDEX = 2;
constexpr uint32_t K_QUERY_ROPE_INDEX = 3;
constexpr uint32_t K_KEY_ROPE_INDEX = 4;
constexpr uint32_t K_SEQ_KV_INDEX = 5;
constexpr uint32_t K_BLOCK_TABLE_INDEX = 6;
constexpr int32_t K_KEY_TENSOR_LIST_INDEX = 1;
constexpr int32_t K_VALUE_TENSOR_LIST_INDEX = 2;
constexpr int32_t K_ACLNN_QUERY_ROPE_INDEX = 24;
constexpr int32_t K_ACLNN_KEY_ROPE_INDEX = 25;
constexpr int32_t K_ACLNN_BLOCK_TABLE_INDEX = 14;
constexpr int32_t K_ACLNN_MASK_INDEX = 4;
constexpr int32_t K_ACLNN_ACTUAL_SEQ_Q_INDEX = 5;
constexpr int32_t K_ACLNN_ACTUAL_SEQ_KV_INDEX = 6;
constexpr uint64_t K_INT32_BYTES = 4;

std::vector<int64_t> BuildActualSeqQLen(
    const std::vector<int32_t> &querySeqLens, int64_t speculativeTokenNum)
{
    std::vector<int64_t> seqLens;
    if (speculativeTokenNum > 1) {
        if (querySeqLens.size() % static_cast<size_t>(speculativeTokenNum) != 0) {
            return {};
        }
        for (int32_t seqLen : querySeqLens) {
            if (seqLen != 1) {
                return {};
            }
        }
        const size_t requestNum =
            querySeqLens.size() / static_cast<size_t>(speculativeTokenNum);
        seqLens.reserve(requestNum);
        for (size_t requestIdx = 0; requestIdx < requestNum; ++requestIdx) {
            seqLens.emplace_back(
                static_cast<int64_t>(requestIdx + 1) * speculativeTokenNum);
        }
        return seqLens;
    }

    seqLens.reserve(querySeqLens.size());
    int64_t cumulative = 0;
    for (int32_t seqLen : querySeqLens) {
        if (seqLen <= 0) {
            return {};
        }
        cumulative += static_cast<int64_t>(seqLen);
        seqLens.emplace_back(cumulative);
    }
    return seqLens;
}

std::vector<int64_t> BuildActualSeqKVLen(
    const std::vector<int32_t> &kvSeqLens, int64_t speculativeTokenNum)
{
    std::vector<int64_t> seqLens;
    if (speculativeTokenNum <= 1) {
        seqLens.reserve(kvSeqLens.size());
        for (int32_t seqLen : kvSeqLens) {
            if (seqLen < 0) {
                return {};
            }
            seqLens.emplace_back(static_cast<int64_t>(seqLen));
        }
        return seqLens;
    }

    if (kvSeqLens.size() % static_cast<size_t>(speculativeTokenNum) != 0) {
        return {};
    }
    const size_t requestNum =
        kvSeqLens.size() / static_cast<size_t>(speculativeTokenNum);
    // Eagle3 supplies one KV length per [step, request] row. FIA needs the
    // final KV length of each request after all validation tokens are cached.
    const size_t finalStepOffset =
        (static_cast<size_t>(speculativeTokenNum) - 1) * requestNum;
    seqLens.reserve(requestNum);
    for (size_t requestIdx = 0; requestIdx < requestNum; ++requestIdx) {
        const int32_t seqLen = kvSeqLens[finalStepOffset + requestIdx];
        if (seqLen < 0) {
            return {};
        }
        seqLens.emplace_back(static_cast<int64_t>(seqLen));
    }
    return seqLens;
}

int64_t InferCacheHeadDim(const atb::Dims &cacheShape, int64_t numKeyValueHeads)
{
    if (cacheShape.dimNum >= 4) {
        return cacheShape.dims[3];
    }
    if (cacheShape.dimNum == 3 && numKeyValueHeads > 0) {
        return cacheShape.dims[2] / numKeyValueHeads;
    }
    return cacheShape.dims[cacheShape.dimNum - 1];
}

int64_t InferCacheBlockSize(const atb::Dims &cacheShape, int64_t numKeyValueHeads)
{
    if (cacheShape.dimNum >= 4) {
        if (cacheShape.dims[1] == numKeyValueHeads) {
            return cacheShape.dims[2];
        }
        if (cacheShape.dims[2] == numKeyValueHeads) {
            return cacheShape.dims[1];
        }
        return cacheShape.dims[2];
    }
    if (cacheShape.dimNum == 3) {
        return cacheShape.dims[1];
    }
    return 0;
}

void SetMlaPageCacheView(const atb::Dims &oldShape, atb::Dims &viewShape)
{
    viewShape = oldShape;
    if (oldShape.dimNum == 4 && oldShape.dims[2] == 1 && oldShape.dims[1] > 1) {
        viewShape.dims[1] = oldShape.dims[2];
        viewShape.dims[2] = oldShape.dims[1];
    }
}

}  // namespace

MlaFusedInferAttentionOperation::MlaFusedInferAttentionOperation(
    const std::string &name,
    const MlaFusedInferAttentionParam &param)
    : AclNNOperation(name), param_(param)
{
    ATB_SPEED_LOG_DEBUG(
        "MlaFusedInferAttentionOperation, param: " << param_.ToString());
}

MlaFusedInferAttentionOperation::~MlaFusedInferAttentionOperation()
{
    ATB_SPEED_LOG_DEBUG("~MlaFusedInferAttentionOperation");
}

uint32_t MlaFusedInferAttentionOperation::GetInputNum() const
{
    uint32_t inputNum = K_REQUIRED_INPUT_NUM;
    if (param_.needMask) {
        inputNum += 1;
    }
    if (param_.needQuerySeqLen) {
        inputNum += 1;
    }
    return inputNum;
}

uint32_t MlaFusedInferAttentionOperation::GetOutputNum() const
{
    return 1;
}

atb::Status MlaFusedInferAttentionOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    outTensorDescs.at(0) = inTensorDescs.at(K_QUERY_INDEX);
    outTensorDescs.at(0).dtype = inTensorDescs.at(K_QUERY_INDEX).dtype;
    const int64_t valueHeadDim =
        param_.valueHeadDim > 0
            ? param_.valueHeadDim
            : InferCacheHeadDim(
                  inTensorDescs.at(K_VALUE_INDEX).shape,
                  param_.numKeyValueHeads);
    if (param_.inputLayout == "TND") {
        outTensorDescs.at(0).shape.dims[
            outTensorDescs.at(0).shape.dimNum - 1] = valueHeadDim;
    } else if (param_.inputLayout == "TND_NTD") {
        outTensorDescs.at(0).shape.dimNum = 3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(K_QUERY_INDEX).shape.dims[1];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(K_QUERY_INDEX).shape.dims[0];
        outTensorDescs.at(0).shape.dims[2] = valueHeadDim;
    } else if (param_.inputLayout == "BNSD_NBSD" ||
               param_.inputLayout == "BSND_NBSD") {
        outTensorDescs.at(0).shape.dimNum = 4;
        outTensorDescs.at(0).shape.dims[0] = param_.numQueryHeads;
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(K_QUERY_INDEX).shape.dims[0];
        outTensorDescs.at(0).shape.dims[2] = 1;
        outTensorDescs.at(0).shape.dims[3] = valueHeadDim;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return atb::NO_ERROR;
}

int MlaFusedInferAttentionOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack start");
    int ret = CreateAclNNInTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " CreateAclNNInTensorVariantPack fail");
        return ret;
    }
    ret = CreateAclNNOutTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " CreateAclNNOutTensorVariantPack fail");
        return ret;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
    return atb::NO_ERROR;
}

int MlaFusedInferAttentionOperation::CreateAclNNInTensorVariantPack(
    const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    const uint32_t inputNum = GetInputNum();
    aclnnVariantPack.aclInTensors.resize(inputNum);

    const uint32_t maskIndex = K_REQUIRED_INPUT_NUM;
    const uint32_t qSeqLenIndex = K_REQUIRED_INPUT_NUM + (param_.needMask ? 1 : 0);
    const int64_t totalTokens = variantPack.inTensors.at(K_QUERY_INDEX).desc.shape.dims[0];

    for (size_t i = 0; i < inputNum; ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);

        if (i == K_SEQ_KV_INDEX || (param_.needQuerySeqLen && i == qSeqLenIndex)) {
            const bool isQuerySeqLen = i == qSeqLenIndex;
            const atb::Tensor &seqTensor = variantPack.inTensors.at(i);
            if (seqTensor.desc.dtype != aclDataType::ACL_INT32) {
                ATB_SPEED_LOG_ERROR(opName_ << " seq tensor index " << i << " dtype is not int32");
                return atb::ERROR_INVALID_PARAM;
            }
            if (seqTensor.hostData == nullptr) {
                ATB_SPEED_LOG_ERROR(opName_ << " seq tensor index " << i << " hostData is null");
                return atb::ERROR_INVALID_PARAM;
            }

            const size_t seqLenNum = seqTensor.dataSize / K_INT32_BYTES;
            const int32_t *hostData =
                static_cast<const int32_t *>(seqTensor.hostData);
            std::vector<int32_t> rawSeqLens(hostData, hostData + seqLenNum);
            aclnnTensor->intArrayHostData.dataOri = rawSeqLens;
            if (isQuerySeqLen) {
                aclnnTensor->intArrayHostData.data = BuildActualSeqQLen(
                    rawSeqLens, param_.speculativeTokenNum);
                if (aclnnTensor->intArrayHostData.data.empty() ||
                    aclnnTensor->intArrayHostData.data.back() != totalTokens) {
                    ATB_SPEED_LOG_ERROR(opName_ << " invalid actual_seq_qlen, totalTokens="
                                                << totalTokens);
                    return atb::ERROR_INVALID_PARAM;
                }
                actualSeqQLen_ = aclnnTensor->intArrayHostData.intArray =
                    aclCreateIntArray(aclnnTensor->intArrayHostData.data.data(),
                                      aclnnTensor->intArrayHostData.data.size());
            } else {
                aclnnTensor->intArrayHostData.data = BuildActualSeqKVLen(
                    rawSeqLens, param_.speculativeTokenNum);
                if (aclnnTensor->intArrayHostData.data.empty()) {
                    ATB_SPEED_LOG_ERROR(opName_ << " invalid actual_seq_kvlen");
                    return atb::ERROR_INVALID_PARAM;
                }
                actualSeqKVLen_ = aclnnTensor->intArrayHostData.intArray =
                    aclCreateIntArray(aclnnTensor->intArrayHostData.data.data(),
                                      aclnnTensor->intArrayHostData.data.size());
            }
            aclnnTensor->intArrayHostData.dataSize = aclnnTensor->intArrayHostData.data.size();
            aclnnTensor->needUpdateTensorDataPtr = false;
            aclnnTensor->tensorIdx = isQuerySeqLen ?
                K_ACLNN_ACTUAL_SEQ_Q_INDEX : K_ACLNN_ACTUAL_SEQ_KV_INDEX;
        } else {
            aclnnTensor->needUpdateTensorDataPtr = true;
            atb::Tensor atbTensor = variantPack.inTensors.at(i);
            atb::Dims viewShape = atbTensor.desc.shape;
            if (i == K_KEY_INDEX || i == K_VALUE_INDEX || i == K_KEY_ROPE_INDEX) {
                SetMlaPageCacheView(atbTensor.desc.shape, viewShape);
            }
            aclnnTensor->strides = GetCopyTensorStride(viewShape);
            aclnnTensor->tensorIdx = static_cast<int>(i);
            if (i == K_KEY_INDEX) {
                aclnnTensor->tensorIdx = 0;
                aclnnTensor->tensorListidx = K_KEY_TENSOR_LIST_INDEX;
            } else if (i == K_VALUE_INDEX) {
                aclnnTensor->tensorIdx = 0;
                aclnnTensor->tensorListidx = K_VALUE_TENSOR_LIST_INDEX;
            } else if (i == K_QUERY_ROPE_INDEX) {
                aclnnTensor->tensorIdx = K_ACLNN_QUERY_ROPE_INDEX;
            } else if (i == K_KEY_ROPE_INDEX) {
                aclnnTensor->tensorIdx = K_ACLNN_KEY_ROPE_INDEX;
            } else if (i == K_BLOCK_TABLE_INDEX) {
                aclnnTensor->tensorIdx = K_ACLNN_BLOCK_TABLE_INDEX;
            } else if (param_.needMask && i == maskIndex) {
                aclnnTensor->tensorIdx = K_ACLNN_MASK_INDEX;
            }
            aclnnTensor->tensor = aclCreateTensor(viewShape.dims,
                                                  viewShape.dimNum,
                                                  atbTensor.desc.dtype,
                                                  aclnnTensor->strides.data(),
                                                  0,
                                                  atbTensor.desc.format,
                                                  atbTensor.desc.shape.dims,
                                                  atbTensor.desc.shape.dimNum,
                                                  atbTensor.deviceData);
            if (aclnnTensor->tensor == nullptr) {
                ATB_SPEED_LOG_ERROR(opName_ << " InTensor index " << i << " create fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }

    if (param_.needQuerySeqLen) {
        const size_t querySeqNum =
            aclnnVariantPack.aclInTensors.at(qSeqLenIndex)
                ->intArrayHostData.data.size();
        const size_t kvSeqNum =
            aclnnVariantPack.aclInTensors.at(K_SEQ_KV_INDEX)
                ->intArrayHostData.data.size();
        if (querySeqNum != kvSeqNum) {
            ATB_SPEED_LOG_ERROR(
                opName_ << " query/kv sequence count mismatch, query="
                        << querySeqNum << ", kv=" << kvSeqNum);
            return atb::ERROR_INVALID_PARAM;
        }
        const atb::Dims &blockTableShape =
            variantPack.inTensors.at(K_BLOCK_TABLE_INDEX).desc.shape;
        if (blockTableShape.dimNum < 2 ||
            blockTableShape.dims[0] != static_cast<int64_t>(querySeqNum)) {
            ATB_SPEED_LOG_ERROR(
                opName_ << " block table row count mismatch, rows="
                        << (blockTableShape.dimNum > 0 ? blockTableShape.dims[0] : 0)
                        << ", query=" << querySeqNum);
            return atb::ERROR_INVALID_PARAM;
        }
    }

    std::vector<aclTensor *> keyList = {
        aclnnVariantPack.aclInTensors.at(K_KEY_INDEX)->tensor};
    std::vector<aclTensor *> valueList = {
        aclnnVariantPack.aclInTensors.at(K_VALUE_INDEX)->tensor};
    aclnnVariantPack.aclInTensorList.clear();
    aclnnVariantPack.aclInTensorList.push_back(nullptr);
    aclnnVariantPack.aclInTensorList.push_back(
        aclCreateTensorList(keyList.data(), keyList.size()));
    aclnnVariantPack.aclInTensorList.push_back(
        aclCreateTensorList(valueList.data(), valueList.size()));
    return atb::NO_ERROR;
}

int MlaFusedInferAttentionOperation::CreateAclNNOutTensorVariantPack(
    const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        aclnnVariantPack.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i), i);
        if (aclnnVariantPack.aclOutTensors[i]->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(opName_ << " OutTensor index " << i << " create fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}

int MlaFusedInferAttentionOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " GetWorkspace start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclTensor *query = aclnnVariantPack.aclInTensors.at(K_QUERY_INDEX)->tensor;
    aclTensor *queryRope = aclnnVariantPack.aclInTensors.at(K_QUERY_ROPE_INDEX)->tensor;
    aclTensor *keyRope = aclnnVariantPack.aclInTensors.at(K_KEY_ROPE_INDEX)->tensor;
    aclTensor *attenMask = nullptr;
    if (param_.needMask) {
        attenMask = aclnnVariantPack.aclInTensors.at(K_REQUIRED_INPUT_NUM)->tensor;
    }
    aclTensor *blockTable =
        aclnnVariantPack.aclInTensors.at(K_BLOCK_TABLE_INDEX)->tensor;
    aclTensor *attentionOut = aclnnVariantPack.aclOutTensors.at(0)->tensor;
    const int64_t blockSize = param_.blockSize > 0 ? param_.blockSize :
        InferCacheBlockSize(
            aclnnVariantPack.aclInTensors.at(K_KEY_INDEX)->atbTensor.desc.shape,
            param_.numKeyValueHeads);

    int ret = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        query,
        aclnnVariantPack.aclInTensorList.at(K_KEY_TENSOR_LIST_INDEX),
        aclnnVariantPack.aclInTensorList.at(K_VALUE_TENSOR_LIST_INDEX),
        nullptr, attenMask,
        actualSeqQLen_, actualSeqKVLen_,
        nullptr, nullptr, nullptr,
        nullptr, nullptr,
        nullptr, nullptr,
        blockTable,
        nullptr, nullptr,
        nullptr, nullptr, nullptr,
        nullptr,
        nullptr, nullptr, nullptr,
        queryRope, keyRope,
        nullptr, nullptr,
        nullptr,
        param_.numQueryHeads, param_.softmaxScale, param_.preTokens, param_.nextTokens,
        const_cast<char *>(param_.inputLayout.c_str()),
        param_.numKeyValueHeads, param_.sparseMode, param_.innerPrecise,
        blockSize, 0, param_.returnSoftmaxLse,
        param_.keyQuantMode, param_.valueQuantMode, param_.queryQuantMode,
        attentionOut, nullptr,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(opName_ << " aclnnFusedInferAttentionScoreV4GetWorkspaceSize failed");
    }
    return ret;
}

int MlaFusedInferAttentionOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    int ret = aclnnFusedInferAttentionScoreV4(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(opName_ << " aclnnFusedInferAttentionScoreV4 execute failed");
    }
    return ret;
}

}  // namespace common
}  // namespace atb_speed
