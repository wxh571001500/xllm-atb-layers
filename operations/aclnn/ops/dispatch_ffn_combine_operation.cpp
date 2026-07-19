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
#include "dispatch_ffn_combine_operation.h"
#include <securec.h>
#include <sstream>
#include <vector>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "operations/aclnn/utils/utils.h"
// aclnn interface installed to custom_xllm_math
#include "aclnnop/aclnn_dispatch_ffn_combine.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(
    void *executor, NnopbaseHcclServerType serverType);

namespace atb_speed {
namespace common {

namespace {
// Input tensor index layout passed from the ATB graph node:
//   0  x                [bs, h]
//   1  weight1          [num_local_experts, h, n*2] gate/up weights → TensorList
//   2  weight2          [num_local_experts, n, h]   down weights → TensorList
//   3  expert_ids       [bs]
//   4  scale1           [num_local_experts, n*2]    → TensorList
//   5  scale2           [num_local_experts, h]      → TensorList
//   6  probs            [bs] (expert routing probabilities)
//   7  x_active_mask    [padded_bs] (optional)
constexpr uint32_t IN_X = 0;
constexpr uint32_t IN_WEIGHT1 = 1;
constexpr uint32_t IN_WEIGHT2 = 2;
constexpr uint32_t IN_EXPERT_IDS = 3;
constexpr uint32_t IN_SCALE1 = 4;
constexpr uint32_t IN_SCALE2 = 5;
constexpr uint32_t IN_PROBS = 6;
constexpr uint32_t IN_ACTIVE_MASK = 7;  // optional
constexpr uint32_t IN_REQUIRED = 7;

// Output tensor index layout:
//   0  output             [bs, h]
//   1  expert_token_nums  [num_local_experts]
constexpr uint32_t OUT_OUTPUT = 0;
constexpr uint32_t OUT_EXPERT_TOKEN_NUMS = 1;
constexpr uint32_t NUM_OUTPUTS = 2;

// Indices into weightTensorVectors_
constexpr size_t LIST_WEIGHT1 = 0;
constexpr size_t LIST_WEIGHT2 = 1;
constexpr size_t LIST_SCALE1 = 2;
constexpr size_t LIST_SCALE2 = 3;
constexpr size_t NUM_WEIGHT_LISTS = 4;
}  // namespace

DispatchFfnCombineOperation::DispatchFfnCombineOperation(
    const std::string &name, DispatchFfnCombineParam param)
    : AclNNOperation(name), param_(param) {}

DispatchFfnCombineOperation::~DispatchFfnCombineOperation() {}

atb::Status DispatchFfnCombineOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    // output[0]: same shape as x input [bs, h]
    const auto &xDesc = inTensorDescs.at(IN_X);

    outTensorDescs.at(OUT_OUTPUT).format = xDesc.format;
    outTensorDescs.at(OUT_OUTPUT).dtype = xDesc.dtype;
    outTensorDescs.at(OUT_OUTPUT).shape.dimNum = 2;
    outTensorDescs.at(OUT_OUTPUT).shape.dims[0] = xDesc.shape.dims[0];  // bs
    outTensorDescs.at(OUT_OUTPUT).shape.dims[1] = xDesc.shape.dims[1];  // h

    // output[1]: expert_token_nums [num_local_experts], dtype = int32
    int64_t numLocalExperts = param_.localMoeExpertNum;
    outTensorDescs.at(OUT_EXPERT_TOKEN_NUMS).format = ACL_FORMAT_ND;
    outTensorDescs.at(OUT_EXPERT_TOKEN_NUMS).dtype = ACL_INT32;
    outTensorDescs.at(OUT_EXPERT_TOKEN_NUMS).shape.dimNum = 1;
    outTensorDescs.at(OUT_EXPERT_TOKEN_NUMS).shape.dims[0] = numLocalExperts;

    return atb::NO_ERROR;
}

uint32_t DispatchFfnCombineOperation::GetInputNum() const
{
    // 7 required + 1 optional (x_active_mask)
    return IN_REQUIRED;  // For now, assume x_active_mask is not used
}

uint32_t DispatchFfnCombineOperation::GetOutputNum() const
{
    return NUM_OUTPUTS;
}

// Override to build both plain tensors and the four weight TensorLists.
atb::Status DispatchFfnCombineOperation::CreateAclNNInTensorVariantPack(
    const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    const uint32_t inputNum = GetInputNum();
    aclnnVariantPack.aclInTensors.resize(inputNum);

    for (uint32_t i = 0; i < inputNum; ++i) {
        aclnnVariantPack.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i), i);
        if (aclnnVariantPack.aclInTensors[i]->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(opName_ << " failed to create aclTensor for input " << i);
            return atb::ERROR_INTERNAL_ERROR;
        }
    }

    // Build four TensorLists for FFN weight inputs (weight1, weight2, scale1, scale2).
    // Each is a single 3D stacked tensor [num_local_experts, ...] wrapped as TensorList length 1.
    weightTensorVectors_.resize(NUM_WEIGHT_LISTS);
    aclnnVariantPack.aclInTensorList.clear();

    // weight1 (gate/up)
    weightTensorVectors_[LIST_WEIGHT1].clear();
    weightTensorVectors_[LIST_WEIGHT1].push_back(aclnnVariantPack.aclInTensors.at(IN_WEIGHT1)->tensor);
    aclnnVariantPack.aclInTensorList.push_back(
        aclCreateTensorList(weightTensorVectors_[LIST_WEIGHT1].data(), weightTensorVectors_[LIST_WEIGHT1].size()));

    // weight2 (down)
    weightTensorVectors_[LIST_WEIGHT2].clear();
    weightTensorVectors_[LIST_WEIGHT2].push_back(aclnnVariantPack.aclInTensors.at(IN_WEIGHT2)->tensor);
    aclnnVariantPack.aclInTensorList.push_back(
        aclCreateTensorList(weightTensorVectors_[LIST_WEIGHT2].data(), weightTensorVectors_[LIST_WEIGHT2].size()));

    // scale1
    weightTensorVectors_[LIST_SCALE1].clear();
    weightTensorVectors_[LIST_SCALE1].push_back(aclnnVariantPack.aclInTensors.at(IN_SCALE1)->tensor);
    aclnnVariantPack.aclInTensorList.push_back(
        aclCreateTensorList(weightTensorVectors_[LIST_SCALE1].data(), weightTensorVectors_[LIST_SCALE1].size()));

    // scale2
    weightTensorVectors_[LIST_SCALE2].clear();
    weightTensorVectors_[LIST_SCALE2].push_back(aclnnVariantPack.aclInTensors.at(IN_SCALE2)->tensor);
    aclnnVariantPack.aclInTensorList.push_back(
        aclCreateTensorList(weightTensorVectors_[LIST_SCALE2].data(), weightTensorVectors_[LIST_SCALE2].size()));

    return atb::NO_ERROR;
}

atb::Status DispatchFfnCombineOperation::CreateAclNNOutTensorVariantPack(
    const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(NUM_OUTPUTS);
    for (uint32_t i = 0; i < NUM_OUTPUTS; ++i) {
        aclnnVariantPack.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i), i);
        if (aclnnVariantPack.aclOutTensors[i]->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(opName_ << " failed to create aclTensor for output " << i);
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}

int DispatchFfnCombineOperation::SetAclNNWorkspaceExecutor()
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    // x_active_mask is optional (nullptr if not provided)
    aclTensor *activeMask = nullptr;  // TODO: support this if needed

    // epCommName must be a C string
    char *groupEpPtr = const_cast<char *>(param_.epCommName.c_str());

    int ret = aclnnDispatchFFNCombineGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(IN_X)->tensor,
        aclnnVariantPack.aclInTensorList.at(LIST_WEIGHT1),
        aclnnVariantPack.aclInTensorList.at(LIST_WEIGHT2),
        aclnnVariantPack.aclInTensors.at(IN_EXPERT_IDS)->tensor,
        aclnnVariantPack.aclInTensorList.at(LIST_SCALE1),
        aclnnVariantPack.aclInTensorList.at(LIST_SCALE2),
        aclnnVariantPack.aclInTensors.at(IN_PROBS)->tensor,
        activeMask,
        groupEpPtr,
        param_.maxOutputSize,
        param_.swigluLimit,
        aclnnVariantPack.aclOutTensors.at(OUT_OUTPUT)->tensor,
        aclnnVariantPack.aclOutTensors.at(OUT_EXPERT_TOKEN_NUMS)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);

    return ret;
}

int DispatchFfnCombineOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    if (NnopbaseSetHcclServerType != nullptr) {
        NnopbaseSetHcclServerType(
            this->aclnnOpCache_->aclExecutor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    int ret = aclnnDispatchFFNCombine(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    return ret;
}

}  // namespace common
}  // namespace atb_speed
