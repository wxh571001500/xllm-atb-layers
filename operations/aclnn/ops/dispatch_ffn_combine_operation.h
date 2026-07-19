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
#ifndef ATB_SPEED_PLUGIN_ACLNN_DISPATCH_FFN_COMBINE_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_DISPATCH_FFN_COMBINE_OPERATION_H
#include <vector>
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

// Fused EP dispatch + FFN + combine for decode (aligns with vLLM MC2 mode=1).
// Wraps aclnnDispatchFFNCombine that performs EP all-to-all communication
// together with gated FFN (gate/up -> swiglu -> down) computation.
struct DispatchFfnCombineParam {
    int64_t epRankSize = 1;
    int64_t epRankId = 0;
    int64_t maxOutputSize = 0;
    double swigluLimit = 0.0;       // swiglu saturation limit (0.0: no limit)
    int64_t localMoeExpertNum = 1;  // number of experts on this rank
    std::string epCommName;
};

class DispatchFfnCombineOperation : public AclNNOperation {
public:
    explicit DispatchFfnCombineOperation(const std::string &name, DispatchFfnCombineParam param);
    ~DispatchFfnCombineOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    DispatchFfnCombineParam param_;
    // Backing storage for the four FFN weight TensorLists:
    //   0: weight1 (gate/up), 1: weight2 (down), 2: scale1, 3: scale2
    std::vector<std::vector<aclTensor *>> weightTensorVectors_;
};

}  // namespace common
}  // namespace atb_speed
#endif
