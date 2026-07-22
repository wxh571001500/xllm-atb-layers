/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "dispatch_gmm_combine_decode_operation.h"

#include <cstdint>
#include <memory>

#include <atb/types.h>

#include "acl/acl.h"
#if __has_include("aclnnop/aclnn_dispatch_gmm_combine_decode.h")
#include "aclnnop/aclnn_dispatch_gmm_combine_decode.h"
#else
#include "aclnn_dispatch_gmm_combine_decode.h"
#endif
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {
namespace {

constexpr uint32_t IN_X = 0;
constexpr uint32_t IN_EXPERT_IDS = 1;
constexpr uint32_t IN_GMM1_WEIGHT = 2;
constexpr uint32_t IN_GMM1_WEIGHT_SCALE = 3;
constexpr uint32_t IN_GMM2_WEIGHT = 4;
constexpr uint32_t IN_GMM2_WEIGHT_SCALE = 5;
constexpr uint32_t IN_EXPERT_SCALES = 6;
constexpr uint32_t IN_PADDING_IDX = 7;
constexpr uint32_t INPUT_NUM = 8;
constexpr uint32_t OUTPUT_NUM = 2;
constexpr uint32_t OUT_HIDDEN_STATES = 0;
constexpr uint32_t OUT_EXPERT_TOKEN_NUMS = 1;
constexpr uint32_t NZ_BLOCK_DIM = 4;

bool IsTensorListInput(uint32_t tensorId)
{
    return tensorId >= IN_GMM1_WEIGHT && tensorId <= IN_GMM2_WEIGHT_SCALE;
}

atb::Dims GetStorageShape(const atb::TensorDesc &tensorDesc)
{
    atb::Dims storageDims = tensorDesc.shape;
    if (tensorDesc.format != ACL_FORMAT_FRACTAL_NZ || tensorDesc.shape.dimNum != DIM3) {
        return storageDims;
    }

    storageDims.dimNum = 5;
    storageDims.dims[DIM0] = tensorDesc.shape.dims[DIM0];
    storageDims.dims[DIM1] = 1 + ((tensorDesc.shape.dims[DIM1] - 1) / 16);
    storageDims.dims[DIM2] = 1 + ((tensorDesc.shape.dims[DIM2] - 1) / 16);
    storageDims.dims[DIM3] = 16;
    storageDims.dims[NZ_BLOCK_DIM] = 16;
    return storageDims;
}

}  // namespace

DispatchGmmCombineDecodeOperation::DispatchGmmCombineDecodeOperation(
    const std::string &name, DispatchGmmCombineDecodeParam param) : AclNNOperation(name), param_(param) {}

DispatchGmmCombineDecodeOperation::~DispatchGmmCombineDecodeOperation() {}

atb::Status DispatchGmmCombineDecodeOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " DispatchGmmCombineDecodeOperation infer shape start");
    outTensorDescs.at(OUT_HIDDEN_STATES) = inTensorDescs.at(IN_X);

    outTensorDescs.at(OUT_EXPERT_TOKEN_NUMS).format = ACL_FORMAT_ND;
    outTensorDescs.at(OUT_EXPERT_TOKEN_NUMS).dtype = ACL_INT64;
    outTensorDescs.at(OUT_EXPERT_TOKEN_NUMS).shape.dimNum = 1;
    outTensorDescs.at(OUT_EXPERT_TOKEN_NUMS).shape.dims[DIM0] = GetLocalExpertNum();
    ATB_SPEED_LOG_DEBUG(opName_ << " DispatchGmmCombineDecodeOperation infer shape end");
    return atb::NO_ERROR;
}

uint32_t DispatchGmmCombineDecodeOperation::GetInputNum() const
{
    return INPUT_NUM;
}

uint32_t DispatchGmmCombineDecodeOperation::GetOutputNum() const
{
    return OUTPUT_NUM;
}

int64_t DispatchGmmCombineDecodeOperation::GetGlobalBS(
    const atb::TensorDesc &xTensorDesc, const atb::TensorDesc &paddingIdxDesc) const
{
    if (param_.globalBS > 0) {
        return param_.globalBS;
    }

    int64_t localTokenNum = param_.maxDecodeDpTokenSize;
    if (localTokenNum == 0 && paddingIdxDesc.shape.dimNum > 0 && paddingIdxDesc.shape.dims[DIM0] > 1) {
        localTokenNum = paddingIdxDesc.shape.dims[DIM0];
    }
    if (localTokenNum == 0 && xTensorDesc.shape.dimNum > 0) {
        localTokenNum = xTensorDesc.shape.dims[DIM0];
    }
    return localTokenNum * param_.epRankSize;
}

int64_t DispatchGmmCombineDecodeOperation::GetLocalExpertNum() const
{
    if (param_.epRankId < param_.sharedExpertRankNum) {
        return 1;
    }
    if (param_.localMoeExpertNum > 0) {
        return param_.localMoeExpertNum;
    }
    int64_t routedEpSize = param_.epRankSize - param_.sharedExpertRankNum;
    if (routedEpSize <= 0) {
        return param_.moeExpertNum;
    }
    return param_.moeExpertNum / routedEpSize;
}

atb::Status DispatchGmmCombineDecodeOperation::CreateAclNNInTensorVariantPack(
    const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    inputTensorListStorage_.clear();
    inputTensorListStorage_.resize(GetInputNum());

    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        aclnnTensor->tensorIdx = static_cast<int>(i);
        aclnnTensor->needUpdateTensorDataPtr = i != IN_PADDING_IDX;
        atb::Tensor atbTensor = variantPack.inTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);

        if (IsTensorListInput(static_cast<uint32_t>(i))) {
            aclnnTensor->tensorListidx = static_cast<int>(i);
            aclnnTensor->tensorIdx = 0;
        }

        atb::Dims storageDims = GetStorageShape(atbTensor.desc);
        CHECK_OPERATION_STATUS_RETURN(
            CallAclCreateTensor(atbTensor.desc.shape, storageDims, atbTensor, aclnnTensor));
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }

    aclnnVariantPack.aclInTensorList.clear();
    aclnnVariantPack.aclInTensorList.resize(IN_GMM2_WEIGHT_SCALE + 1, nullptr);
    for (uint32_t i = IN_GMM1_WEIGHT; i <= IN_GMM2_WEIGHT_SCALE; ++i) {
        inputTensorListStorage_.at(i).clear();
        inputTensorListStorage_.at(i).push_back(aclnnVariantPack.aclInTensors.at(i)->tensor);
        aclnnVariantPack.aclInTensorList.at(i) = aclCreateTensorList(
            inputTensorListStorage_.at(i).data(), inputTensorListStorage_.at(i).size());
    }
    return atb::NO_ERROR;
}

int DispatchGmmCombineDecodeOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " DispatchGmmCombineDecodeOperation start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int64_t globalBS = GetGlobalBS(
        aclnnVariantPack.aclInTensors.at(IN_X)->atbTensor.desc,
        aclnnVariantPack.aclInTensors.at(IN_PADDING_IDX)->atbTensor.desc);

    int ret = aclnnDispatchGmmCombineDecodeGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(IN_X)->tensor,
        aclnnVariantPack.aclInTensors.at(IN_EXPERT_IDS)->tensor,
        aclnnVariantPack.aclInTensorList.at(IN_GMM1_WEIGHT),
        aclnnVariantPack.aclInTensorList.at(IN_GMM1_WEIGHT_SCALE),
        aclnnVariantPack.aclInTensorList.at(IN_GMM2_WEIGHT),
        aclnnVariantPack.aclInTensorList.at(IN_GMM2_WEIGHT_SCALE),
        aclnnVariantPack.aclInTensors.at(IN_EXPERT_SCALES)->tensor,
        nullptr,
        nullptr,
        param_.epCommName.data(),
        param_.epRankSize,
        param_.epRankId,
        param_.moeExpertNum,
        param_.sharedExpertNum,
        param_.sharedExpertRankNum,
        param_.quantMode,
        globalBS,
        aclnnVariantPack.aclOutTensors.at(OUT_HIDDEN_STATES)->tensor,
        aclnnVariantPack.aclOutTensors.at(OUT_EXPERT_TOKEN_NUMS)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                                << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                                << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int DispatchGmmCombineDecodeOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " DispatchGmmCombineDecodeOperation execute start");
    int ret = aclnnDispatchGmmCombineDecode(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " DispatchGmmCombineDecodeOperation execute end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed
