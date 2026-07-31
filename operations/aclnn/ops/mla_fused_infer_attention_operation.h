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
#ifndef ATB_SPEED_PLUGIN_ACLNN_MLA_FUSED_INFER_ATTENTION_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MLA_FUSED_INFER_ATTENTION_OPERATION_H

#include <sstream>
#include <string>

#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

struct MlaFusedInferAttentionParam {
    bool needMask = false;
    bool needQuerySeqLen = true;
    int64_t numQueryHeads = 0;
    int64_t numKeyValueHeads = 1;
    int64_t valueHeadDim = 0;
    int64_t speculativeTokenNum = 1;
    double softmaxScale = 1.0;
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
    std::string inputLayout = "TND_NTD";
    int64_t sparseMode = 3;
    int64_t innerPrecise = 0;
    int64_t blockSize = 0;
    int64_t queryQuantMode = 0;
    int64_t keyQuantMode = 0;
    int64_t valueQuantMode = 0;
    bool returnSoftmaxLse = false;

    std::string ToString() const
    {
        std::ostringstream oss;
        oss << "MlaFusedInferAttentionParam {" << std::endl;
        oss << "  needMask: " << needMask << std::endl;
        oss << "  needQuerySeqLen: " << needQuerySeqLen << std::endl;
        oss << "  numQueryHeads: " << numQueryHeads << std::endl;
        oss << "  numKeyValueHeads: " << numKeyValueHeads << std::endl;
        oss << "  valueHeadDim: " << valueHeadDim << std::endl;
        oss << "  speculativeTokenNum: " << speculativeTokenNum << std::endl;
        oss << "  softmaxScale: " << softmaxScale << std::endl;
        oss << "  preTokens: " << preTokens << std::endl;
        oss << "  nextTokens: " << nextTokens << std::endl;
        oss << "  inputLayout: " << inputLayout << std::endl;
        oss << "  sparseMode: " << sparseMode << std::endl;
        oss << "  innerPrecise: " << innerPrecise << std::endl;
        oss << "  blockSize: " << blockSize << std::endl;
        oss << "  queryQuantMode: " << queryQuantMode << std::endl;
        oss << "  keyQuantMode: " << keyQuantMode << std::endl;
        oss << "  valueQuantMode: " << valueQuantMode << std::endl;
        oss << "  returnSoftmaxLse: " << returnSoftmaxLse << std::endl;
        oss << "}";
        return oss.str();
    }
};

class MlaFusedInferAttentionOperation : public AclNNOperation {
public:
    explicit MlaFusedInferAttentionOperation(
        const std::string &name, const MlaFusedInferAttentionParam &param);
    ~MlaFusedInferAttentionOperation() override;

    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;

protected:
    int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

private:
    MlaFusedInferAttentionParam param_;
    aclIntArray *actualSeqQLen_ = nullptr;
    aclIntArray *actualSeqKVLen_ = nullptr;
};

}  // namespace common
}  // namespace atb_speed

#endif
