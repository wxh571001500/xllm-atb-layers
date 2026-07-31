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
#include "mla_transpose_batch_matmul_operation.h"

#include <array>
#include <memory>

#include "acl/acl.h"
#include "aclnnop/aclnn_transpose_batch_mat_mul.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {
namespace {

constexpr uint32_t K_INPUT_INDEX = 0;
constexpr uint32_t K_WEIGHT_INDEX = 1;
constexpr uint32_t K_OUTPUT_INDEX = 0;
constexpr int32_t K_BATCH_SPLIT_FACTOR = 1;

int64_t GetElementCount(const atb::Dims &shape)
{
    int64_t elementCount = 1;
    for (uint32_t i = 0; i < shape.dimNum; ++i) {
        elementCount *= shape.dims[i];
    }
    return elementCount;
}

}  // namespace

MlaTransposeBatchMatMulOperation::MlaTransposeBatchMatMulOperation(
    const std::string &name, const MlaTransposeBatchMatMulParam &param)
    : AclNNOperation(name), param_(param)
{
}

MlaTransposeBatchMatMulOperation::~MlaTransposeBatchMatMulOperation()
{
    if (permX1_ != nullptr) {
        aclDestroyIntArray(permX1_);
    }
    if (permX2_ != nullptr) {
        aclDestroyIntArray(permX2_);
    }
    if (permY_ != nullptr) {
        aclDestroyIntArray(permY_);
    }
    this->DestroyOperation();
}

uint32_t MlaTransposeBatchMatMulOperation::GetInputNum() const
{
    return NUM2;
}

uint32_t MlaTransposeBatchMatMulOperation::GetOutputNum() const
{
    return NUM1;
}

atb::Status MlaTransposeBatchMatMulOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    const int64_t headElements = param_.numHeads * param_.headDim;
    const int64_t inputElements = GetElementCount(inTensorDescs.at(K_INPUT_INDEX).shape);
    const atb::Dims &weightShape = inTensorDescs.at(K_WEIGHT_INDEX).shape;
    if (headElements <= 0 || inputElements % headElements != 0 ||
        weightShape.dimNum != NUM3 || weightShape.dims[0] != param_.numHeads ||
        weightShape.dims[1] != param_.headDim) {
        ATB_SPEED_LOG_ERROR(opName_ << " invalid MLA transpose batch matmul shape");
        return atb::ERROR_INVALID_PARAM;
    }

    outTensorDescs.at(K_OUTPUT_INDEX).format = ACL_FORMAT_ND;
    outTensorDescs.at(K_OUTPUT_INDEX).dtype = inTensorDescs.at(K_INPUT_INDEX).dtype;
    outTensorDescs.at(K_OUTPUT_INDEX).shape.dimNum = NUM3;
    outTensorDescs.at(K_OUTPUT_INDEX).shape.dims[0] = inputElements / headElements;
    outTensorDescs.at(K_OUTPUT_INDEX).shape.dims[1] = param_.numHeads;
    outTensorDescs.at(K_OUTPUT_INDEX).shape.dims[2] =
        weightShape.dims[2];
    return atb::NO_ERROR;
}

atb::Status MlaTransposeBatchMatMulOperation::CreateAclNNInTensorVariantPack(
    const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (uint32_t i = 0; i < GetInputNum(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = static_cast<int>(i);
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor atbTensor = variantPack.inTensors.at(i);
        atb::Dims viewShape = atbTensor.desc.shape;
        if (i == K_INPUT_INDEX) {
            const int64_t headElements = param_.numHeads * param_.headDim;
            const int64_t inputElements = GetElementCount(atbTensor.desc.shape);
            if (headElements <= 0 || inputElements % headElements != 0) {
                ATB_SPEED_LOG_ERROR(opName_ << " cannot view FIA output as [N, B, L]");
                return atb::ERROR_INVALID_PARAM;
            }
            viewShape.dimNum = NUM3;
            viewShape.dims[0] = param_.numHeads;
            viewShape.dims[1] = inputElements / headElements;
            viewShape.dims[2] = param_.headDim;
        }
        aclnnTensor->strides = GetCopyTensorStride(viewShape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(
            viewShape, atbTensor.desc.shape, atbTensor, aclnnTensor));
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status MlaTransposeBatchMatMulOperation::CreateAclNNOutTensorVariantPack(
    const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(
        variantPack.outTensors.at(K_OUTPUT_INDEX), K_OUTPUT_INDEX);
    if (aclnnTensor->tensor == nullptr) {
        ATB_SPEED_LOG_ERROR(opName_ << " create output tensor failed");
        return atb::ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack.aclOutTensors[K_OUTPUT_INDEX] = aclnnTensor;
    return atb::NO_ERROR;
}

int MlaTransposeBatchMatMulOperation::SetAclNNWorkspaceExecutor()
{
    const std::array<int64_t, NUM3> permX = {0, 1, 2};
    const std::array<int64_t, NUM3> permY = {1, 0, 2};
    if (permX1_ == nullptr) {
        permX1_ = aclCreateIntArray(permX.data(), permX.size());
    }
    if (permX2_ == nullptr) {
        permX2_ = aclCreateIntArray(permX.data(), permX.size());
    }
    if (permY_ == nullptr) {
        permY_ = aclCreateIntArray(permY.data(), permY.size());
    }
    if (permX1_ == nullptr || permX2_ == nullptr || permY_ == nullptr) {
        ATB_SPEED_LOG_ERROR(opName_ << " create transpose permutation failed");
        return atb::ERROR_INTERNAL_ERROR;
    }

    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    return aclnnTransposeBatchMatMulGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(K_INPUT_INDEX)->tensor,
        aclnnVariantPack.aclInTensors.at(K_WEIGHT_INDEX)->tensor,
        nullptr, nullptr, permX1_, permX2_, permY_, 0, K_BATCH_SPLIT_FACTOR,
        aclnnVariantPack.aclOutTensors.at(K_OUTPUT_INDEX)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
}

int MlaTransposeBatchMatMulOperation::ExecuteAclNNOp(
    uint8_t *workspace, aclrtStream &stream)
{
    return aclnnTransposeBatchMatMul(
        workspace, this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor, stream);
}

}  // namespace common
}  // namespace atb_speed
