/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "moe_gating_topk_operation.h"

#include "acl/acl.h"
#if __has_include("aclnnop/aclnn_moe_gating_top_k.h")
#include "aclnnop/aclnn_moe_gating_top_k.h"
#else
#include "aclnn_moe_gating_top_k.h"
#endif
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

MoeGatingTopKOperation::MoeGatingTopKOperation(
    const std::string &name, MoeGatingTopKParam param) : AclNNOperation(name), param_(param) {}

MoeGatingTopKOperation::~MoeGatingTopKOperation() {}

atb::Status MoeGatingTopKOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeGatingTopKOperation infer shape start");

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = DIM2;
    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM0).shape.dims[DIM1] = param_.k;

    outTensorDescs.at(DIM1).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM1).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(DIM1).shape.dimNum = DIM2;
    outTensorDescs.at(DIM1).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM1).shape.dims[DIM1] = param_.k;

    outTensorDescs.at(DIM2).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM2).dtype = aclDataType::ACL_FLOAT;
    outTensorDescs.at(DIM2).shape.dimNum = DIM2;
    outTensorDescs.at(DIM2).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM2).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];

    ATB_SPEED_LOG_DEBUG(opName_ << "MoeGatingTopKOperation infer shape end");
    return atb::NO_ERROR;
}

uint32_t MoeGatingTopKOperation::GetInputNum() const
{
    return DIM2;
}

uint32_t MoeGatingTopKOperation::GetOutputNum() const
{
    return DIM3;
}

int MoeGatingTopKOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeGatingTopKOperation start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnMoeGatingTopKGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM1)->tensor,
        param_.k,
        param_.kGroup,
        param_.groupCount,
        param_.groupSelectMode,
        param_.renorm,
        param_.normType,
        param_.outFlag,
        param_.routedScalingFactor,
        param_.eps,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM1)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM2)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MoeGatingTopKOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeGatingTopKOperation start");
    int ret = aclnnMoeGatingTopK(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeGatingTopKOperation end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed
