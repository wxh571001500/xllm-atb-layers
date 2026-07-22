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

#ifndef ATB_SPEED_PLUGIN_ACLNN_DISPATCH_GMM_COMBINE_DECODE_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_DISPATCH_GMM_COMBINE_DECODE_OPERATION_H

#include <cstdint>
#include <string>
#include <vector>

#include "atb_speed/utils/operation_util.h"
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"

namespace atb_speed {
namespace common {

struct DispatchGmmCombineDecodeParam {
    int32_t epRankId = 0;
    int32_t epRankSize = 1;
    int32_t maxDecodeDpTokenSize = 0;
    int64_t moeExpertNum = 1;
    int64_t localMoeExpertNum = 1;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t quantMode = 0;
    int64_t globalBS = 0;
    std::string epCommName = "";
};

class DispatchGmmCombineDecodeOperation : public AclNNOperation {
public:
    explicit DispatchGmmCombineDecodeOperation(const std::string &name, DispatchGmmCombineDecodeParam param);
    ~DispatchGmmCombineDecodeOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int64_t GetGlobalBS(const atb::TensorDesc &xTensorDesc, const atb::TensorDesc &paddingIdxDesc) const;
    int64_t GetLocalExpertNum() const;

    DispatchGmmCombineDecodeParam param_;
    std::vector<std::vector<aclTensor *>> inputTensorListStorage_;
};

}  // namespace common
}  // namespace atb_speed

#endif  // ATB_SPEED_PLUGIN_ACLNN_DISPATCH_GMM_COMBINE_DECODE_OPERATION_H
