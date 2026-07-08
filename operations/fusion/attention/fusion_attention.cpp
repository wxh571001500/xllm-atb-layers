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
#include "operations/fusion/attention/fusion_attention.h"
#include <cstdlib>
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/ops/add_rms_norm_operation.h"
#include "operations/aclnn/ops/cast_operation.h"
#include "operations/aclnn/ops/dequant_rope_quant_kvcache_operation.h"
#include "operations/fusion/attention/qkv_linear_split.h"
#include "operations/fusion/attention/self_attention.h"
#include "operations/fusion/infer_shape_functions.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace common {

namespace {

template <typename NormParamType>
bool DebugStopOneRecCrossLayer0AfterQkv(
    const FusionAttentionParam<NormParamType> &param) {
  const char *flag =
      std::getenv("XLLM_DEBUG_ONEREC_CROSS_LAYER0_STOP_AFTER_QKV");
  if (flag == nullptr || std::string(flag) != "1") {
    return false;
  }
  return param.isOneRecCrossAttention && param.isPrefill && param.layerId == 0;
}

template <typename NormParamType>
bool DebugStopOneRecCrossLayer0AfterKvOnly(
    const FusionAttentionParam<NormParamType> &param) {
  const char *flag =
      std::getenv("XLLM_DEBUG_ONEREC_CROSS_LAYER0_STOP_AFTER_KV_ONLY");
  if (flag == nullptr || std::string(flag) != "1") {
    return false;
  }
  return param.isOneRecCrossAttention && param.isPrefill && param.layerId == 0;
}

template <typename NormParamType>
bool DebugStopOneRecCrossLayer0AfterKOnly(
    const FusionAttentionParam<NormParamType> &param) {
  const char *flag =
      std::getenv("XLLM_DEBUG_ONEREC_CROSS_LAYER0_STOP_AFTER_K_ONLY");
  if (flag == nullptr || std::string(flag) != "1") {
    return false;
  }
  return param.isOneRecCrossAttention && param.isPrefill && param.layerId == 0;
}

template <typename NormParamType>
bool DebugStopOneRecCrossLayer0WithIdentityK(
    const FusionAttentionParam<NormParamType> &param) {
  const char *flag =
      std::getenv("XLLM_DEBUG_ONEREC_CROSS_LAYER0_STOP_WITH_IDENTITY_K");
  if (flag == nullptr || std::string(flag) != "1") {
    return false;
  }
  return param.isOneRecCrossAttention && param.isPrefill && param.layerId == 0;
}

template <typename NormParamType>
bool DebugStopOneRecCrossLayer0WithIdentityV(
    const FusionAttentionParam<NormParamType> &param) {
  const char *flag =
      std::getenv("XLLM_DEBUG_ONEREC_CROSS_LAYER0_STOP_WITH_IDENTITY_V");
  if (flag == nullptr || std::string(flag) != "1") {
    return false;
  }
  return param.isOneRecCrossAttention && param.isPrefill && param.layerId == 0;
}

template <typename NormParamType>
bool DebugStopOneRecCrossLayer0WithIdentityKv(
    const FusionAttentionParam<NormParamType> &param) {
  const char *flag =
      std::getenv("XLLM_DEBUG_ONEREC_CROSS_LAYER0_STOP_WITH_IDENTITY_KV");
  if (flag == nullptr || std::string(flag) != "1") {
    return false;
  }
  return param.isOneRecCrossAttention && param.isPrefill && param.layerId == 0;
}

} // namespace

std::map<std::string, std::vector<std::string>> GetAttnInTensorCandidates();
std::map<std::string, std::vector<std::string>>
GetAttnIntermediateTensorCandidates();
std::map<std::string, std::vector<std::string>>
GetAttnIntermediateTensorCandidatesPrefill();

std::vector<std::string> GetOneRecEncoderTensorNames() {
  return {"in_input",
          "in_norm_weight",
          "in_norm_bias",
          "in_norm_new_weight",
          "in_norm_new_bias",
          "in_weight_0",
          "in_scale_0",
          "in_offset_0",
          "in_descale_0",
          "in_bias_0",
          "in_compress_idx_0",
          "in_weight_1",
          "in_scale_1",
          "in_offset_1",
          "in_descale_1",
          "in_bias_1",
          "in_compress_idx_1",
          "in_weight_2",
          "in_scale_2",
          "in_offset_2",
          "in_descale_2",
          "in_bias_2",
          "in_compress_idx_2",
          "in_attention_mask",
          "in_seq_len",
          "in_weight_dense",
          "in_scale_dense",
          "in_offset_dense",
          "in_descale_dense",
          "in_bias_dense",
          "in_compress_idx_dense"};
}

std::vector<std::string> GetOneRecDecoderTensorNames() {
  return {"in_input",
          "in_norm_weight",
          "in_norm_bias",
          "in_norm_new_weight",
          "in_norm_new_bias",
          "in_weight_0",
          "in_scale_0",
          "in_offset_0",
          "in_descale_0",
          "in_bias_0",
          "in_compress_idx_0",
          "in_weight_1",
          "in_scale_1",
          "in_offset_1",
          "in_descale_1",
          "in_bias_1",
          "in_compress_idx_1",
          "in_weight_2",
          "in_scale_2",
          "in_offset_2",
          "in_descale_2",
          "in_bias_2",
          "in_compress_idx_2",
          "in_weight_dense",
          "in_scale_dense",
          "in_offset_dense",
          "in_descale_dense",
          "in_bias_dense",
          "in_compress_idx_dense",
          "in_attention_mask",
          "in_k_cache",
          "in_v_cache",
          "in_seq_len",
          "in_token_offset",
          "in_layer_id",
          "in_block_tables",
          "in_slots_in_pa_or_logn_in_fa"};
}

template <typename NormParamType>
std::map<std::string, uint32_t>
ConstructOneRecTensorMap(const FusionAttentionParam<NormParamType> &param,
                         uint32_t &inTensorNum, uint32_t &outTensorNum,
                         uint32_t &internalTensorNum, bool isEncoder) {
  auto attnInTensorCandidates = GetAttnInTensorCandidates();
  std::map<std::string, std::vector<std::string>>
      attnIntermediateTensorCandidates;
  if (param.isPrefill && param.enableXattention &&
      !param.isOneRecCrossAttention) {
    attnIntermediateTensorCandidates =
        GetAttnIntermediateTensorCandidatesPrefill();
  } else {
    attnIntermediateTensorCandidates = GetAttnIntermediateTensorCandidates();
  }

  std::vector<std::string> inTensorList =
      isEncoder ? GetOneRecEncoderTensorNames() : GetOneRecDecoderTensorNames();
  std::vector<std::string> intermediateTensorList = {};
  std::vector<std::string> outTensorList = {"out"};

  if (param.enableAddNorm) {
    AddTensorToList(attnInTensorCandidates, "add_rmsnorm_quant", inTensorList);
  }
  AddTensorToList(attnIntermediateTensorCandidates, "default",
                  intermediateTensorList);
  // OneRec decoder FAS needs intermediate mask tensor and softmax stats.
  const bool useOneRecDecoderFAS =
      param.isOneRecDecoder &&
      param.attnBackend == atb_speed::common::OpBackend::ACLNN &&
      param.aclnnFAScoreParam.headNum > 0 &&
      !(param.isPrefill && param.enableXattention);
  if (useOneRecDecoderFAS) {
    intermediateTensorList.push_back("intermediate_mask");
    intermediateTensorList.push_back("softmax_max_out");
    intermediateTensorList.push_back("softmax_sum_out");
    intermediateTensorList.push_back("softmax_out_out");
  }

  if (param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
      param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
      param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
    AddTensorToList(attnInTensorCandidates, "alibi_mask_compress",
                    inTensorList);
  }

  if (param.pageAttentionParam.compressType ==
      atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD) {
    AddTensorToList(attnInTensorCandidates, "compress_head_alibi",
                    inTensorList);
  } else if (param.pageAttentionParam.compressType ==
             atb::infer::PagedAttentionParam::CompressType::
                 COMPRESS_TYPE_KVHEAD_ROPE) {
    AddTensorToList(attnInTensorCandidates, "compress_head_rope", inTensorList);
  }
  if (param.pageAttentionParam.calcType ==
          atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC ||
      param.isPrefixCacheWithoutChunk) {
    AddTensorToList(attnInTensorCandidates, "speculate", inTensorList);
  }

  ConstructAttentionQuantTensorMap(param, attnInTensorCandidates,
                                   attnIntermediateTensorCandidates,
                                   inTensorList, intermediateTensorList);

  if (param.supportLora) {
    if (param.useImMask) {
      AddTensorToList(attnInTensorCandidates, "lora_with_mask", inTensorList);
    }
    AddTensorToList(attnInTensorCandidates, "lora", inTensorList);
  }
  if (param.selfOutLinearTensorParallelInfo.quantType !=
      atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
    AddTensorToList(attnInTensorCandidates, "reduce_quant", inTensorList);
  }
  if (param.pageAttentionParam.scaleType ==
      atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN) {
    AddTensorToList(attnInTensorCandidates, "log_n_scale", inTensorList);
  }
  if (param.useQKNorm) {
    AddTensorToList(attnInTensorCandidates, "qk_norm", inTensorList);
  }
  if (param.enableAddNorm) {
    AddTensorToList(attnInTensorCandidates, "add_norm", inTensorList);
    outTensorList.push_back("out_add");
  }
  if (param.enablePreFetchWeight) {
    AddTensorToList(attnInTensorCandidates, "cmo_mlp_first_matmul_weight",
                    inTensorList);
  }
  if (param.enableRopeQuantKvcache) {
    AddTensorToList(attnIntermediateTensorCandidates, "dequant_rope",
                    intermediateTensorList);
  }
  if (param.enableFlashComm) {
    AddTensorToList(attnInTensorCandidates, "flash_comm", inTensorList);
  }
  if (param.enableXattention) {
    AddTensorToList(attnInTensorCandidates, "x_attention", inTensorList);
  }
  if (!param.isPrefill && param.enableAclGraphPagedAttention) {
    AddTensorToList(attnInTensorCandidates, "acl_graph", inTensorList);
  }

  inTensorNum = inTensorList.size();
  outTensorNum = outTensorList.size();
  internalTensorNum = intermediateTensorList.size();

  return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

template <typename NormParamType>
std::map<std::string, uint32_t> ConstructOneRecCrossAttentionTensorMap(
    const FusionAttentionParam<NormParamType> &param, uint32_t &inTensorNum,
    uint32_t &outTensorNum, uint32_t &internalTensorNum) {
  auto attnInTensorCandidates = GetAttnInTensorCandidates();
  auto attnIntermediateTensorCandidates = GetAttnIntermediateTensorCandidates();
  const bool isOneRecCrossPrefill =
      param.isPrefill && param.isOneRecCrossAttention;
  const bool stop_after_qkv = DebugStopOneRecCrossLayer0AfterQkv(param);
  const bool stop_after_kv_only = DebugStopOneRecCrossLayer0AfterKvOnly(param);
  const bool stop_after_k_only = DebugStopOneRecCrossLayer0AfterKOnly(param);
  const bool stop_with_identity_k =
      DebugStopOneRecCrossLayer0WithIdentityK(param);
  const bool stop_with_identity_v =
      DebugStopOneRecCrossLayer0WithIdentityV(param);
  const bool stop_with_identity_kv =
      DebugStopOneRecCrossLayer0WithIdentityKv(param);
  const bool minimizeOneRecCrossAttnInputs =
      param.isOneRecCrossAttention &&
      param.attnBackend == atb_speed::common::OpBackend::ACLNN &&
      param.enableOneRecPrefillOnly && !param.enableSplitFuse && !param.isFA;

  std::vector<std::string> inTensorList = {
      "intermediate_self_residual_out",
      "in_cross_attn_norm_weight",
      "in_cross_attn_norm_bias",
      "in_cross_attn_norm_new_weight",
      "in_cross_attn_norm_new_bias",
      "in_cross_weight_0",
      "in_cross_bias_0",
      "in_cross_descale_0",
      "in_cross_offset_0",
      "in_cross_scale_0",
      "in_cross_compress_idx_0",
      "in_cross_weight_1",
      "in_cross_bias_1",
      "in_cross_descale_1",
      "in_cross_offset_1",
      "in_cross_scale_1",
      "in_cross_compress_idx_1",
      "in_cross_weight_2",
      "in_cross_bias_2",
      "in_cross_descale_2",
      "in_cross_offset_2",
      "in_cross_scale_2",
      "in_cross_compress_idx_2",
      "in_cross_dense_weight",
      "in_cross_dense_bias",
      "in_cross_dense_descale",
      "in_cross_dense_offset",
      "in_cross_dense_scale",
      "in_cross_dense_compress_idx",
      "in_seq_len",
      "in_layer_id",
  };
  if (param.isPrefill) {
    std::vector<std::string> stageSpecific;
    if (minimizeOneRecCrossAttnInputs) {
      stageSpecific = {
          "in_encoder_output",          "in_k_cache",   "in_v_cache",
          "intermediate_self_attn_out", "cross_kv_len",
      };
    } else {
      stageSpecific = {
          "in_encoder_output",
          "in_k_cache",
          "in_v_cache",
          "in_cross_attn_slots",
          "in_cross_attn_seq_len",
          "in_cross_attn_block_tables",
          "intermediate_self_attn_out",
          "cross_kv_len",
      };
    }
    inTensorList.insert(inTensorList.end(), stageSpecific.begin(),
                        stageSpecific.end());
  } else {
    std::vector<std::string> stageSpecific;
    if (minimizeOneRecCrossAttnInputs) {
      stageSpecific = {
          "in_attention_mask",
          "in_cross_k_cache",
          "in_cross_v_cache",
          "cross_kv_len",
      };
    } else {
      stageSpecific = {
          "in_attention_mask",          "in_cross_attn_seq_len",
          "in_cross_attn_block_tables", "in_cross_k_cache",
          "in_cross_v_cache",           "cross_kv_len",
      };
    }
    inTensorList.insert(inTensorList.end(), stageSpecific.begin(),
                        stageSpecific.end());
  }

  std::vector<std::string> intermediateTensorList =
      (stop_after_k_only || stop_after_kv_only || stop_with_identity_k ||
       stop_with_identity_v || stop_with_identity_kv)
          ? std::vector<std::string>{}
          : stop_after_qkv
          ? std::vector<std::string>{"intermediate_q"}
          : ((!param.isPrefill || isOneRecCrossPrefill)
                 ? std::vector<std::string>{"intermediate_q",
                                            "intermediate_self_attention"}
                 : std::vector<std::string>{"intermediate_q",
                                            "intermediate_k",
                                            "intermediate_v",
                                            "intermediate_self_attention"});
  std::vector<std::string> outTensorList = {"out"};

  ConstructAttentionQuantTensorMap(param, attnInTensorCandidates,
                                   attnIntermediateTensorCandidates,
                                   inTensorList, intermediateTensorList);
  if (param.attnBackend == atb_speed::common::OpBackend::ACLNN &&
      !stop_after_qkv && !stop_after_kv_only && !stop_after_k_only &&
      !stop_with_identity_k && !stop_with_identity_v &&
      !stop_with_identity_kv) {
    // Align with xllm_rec: cross-attention only needs one lightweight
    // internal sink for the bs probe node. Reusing the full FIA intermediate
    // set here leaves unassigned tensors in the graph because the cross-attn
    // path never produces intermediate_self_attention_bsnd.
    intermediateTensorList.push_back("intermediate_q_bsnd");
  }

  if (param.supportLora) {
    if (param.useImMask) {
      AddTensorToList(attnInTensorCandidates, "lora_with_mask", inTensorList);
    }
    AddTensorToList(attnInTensorCandidates, "lora", inTensorList);
  }
  if (param.enableAddNorm) {
    AddTensorToList(attnInTensorCandidates, "add_norm", inTensorList);
    outTensorList.push_back("out_add");
  }
  if (isOneRecCrossPrefill) {
    outTensorList.push_back("in_cross_k_cache");
    outTensorList.push_back("in_cross_v_cache");
  }

  inTensorNum = inTensorList.size();
  outTensorNum = outTensorList.size();
  internalTensorNum = intermediateTensorList.size();
  return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

std::map<std::string, std::vector<std::string>> GetAttnInTensorCandidates() {
  std::map<std::string, std::vector<std::string>> attnInTensorCandidates = {
      {"default", {"in_input",
                   "in_norm_weight",
                   "in_norm_bias",
                   "in_norm_new_weight",
                   "in_norm_new_bias",
                   "in_weight_0",
                   "in_scale_0",
                   "in_offset_0",
                   "in_descale_0",
                   "in_bias_0",
                   "in_compress_idx_0",
                   "in_weight_1",
                   "in_scale_1",
                   "in_offset_1",
                   "in_descale_1",
                   "in_bias_1",
                   "in_compress_idx_1",
                   "in_weight_2",
                   "in_scale_2",
                   "in_offset_2",
                   "in_descale_2",
                   "in_bias_2",
                   "in_compress_idx_2",
                   "in_cos_embed",
                   "in_sin_embed",
                   "in_seq_len",
                   "in_k_cache",
                   "in_v_cache",
                   "in_attention_mask",
                   "in_token_offset",
                   "in_layer_id",
                   "in_block_tables",
                   "in_slots_in_pa_or_logn_in_fa",
                   "in_weight_dense",
                   "in_scale_dense",
                   "in_offset_dense",
                   "in_descale_dense",
                   "in_bias_dense",
                   "in_compress_idx_dense"}},
      {"alibi_mask_compress", {"in_slopes"}},
      {"compress_head_alibi", {"in_batch_wins", "in_ra_seq_len"}},
      {"compress_head_rope",
       {"in_batch_wins", "in_ra_seq_len", "in_pffset_index", "in_ra_offset",
        "in_reshape_seq_len"}},
      {"speculate", {"in_q_len"}},
      {"kv_quant_scale",
       {"in_k_quant_scale", "in_k_dequant_scale", "in_v_quant_scale",
        "in_v_dequant_scale"}},
      {"kv_quant_offset",
       {"in_k_quant_offset", "in_k_dequant_offset", "in_v_quant_offset",
        "in_v_dequant_offset"}},
      {"fa3_quant",
       {"in_q_quant_scale", "in_k_quant_scale", "in_v_quant_scale",
        "in_qk_descale", "q_offset", "kv_offset", "fa3_v_quant_scale",
        "fa3_offset"}},
      {"reduce_quant",
       {"in_reduce_quant_scale", "in_reduce_quant_offset",
        "in_gather_quant_scale", "in_gather_quant_offset"}},
      {"lora",
       {"in_seq_len_cum_sum", "in_lora_a_0", "in_lora_b_0", "in_lora_a_1",
        "in_lora_b_1", "in_lora_a_2", "in_lora_b_2", "in_dense_lora_a",
        "in_dense_lora_b"}},
      {"lora_with_mask", {"in_im_mask"}},
      {"log_n_scale", {"in_log_n_scale"}},
      {"qk_norm", {"in_q_norm_weight", "in_k_norm_weight"}},
      {"add_norm", {"in_residual_add"}},
      {"add_rmsnorm_quant", {"in_qkv_scale_fill", "in_qkv_offset_fill"}},
      {"cmo_mlp_first_matmul_weight", {"in_mlp_weight_0"}},
      {"flash_comm",
       {"send_counts", "sdispls", "send_count", "recv_counts", "rdispls",
        "recv_count", "fake_rs_shape", "fake_ag_shape"}},
      {"acl_graph", {"paged_attention_tiling_data"}},
      {"x_attention",
       {"in_decode_k_cache", "in_decode_v_cache", "in_beam_width",
        "in_current_round"}},
      {"fia", {"fia_padding_idx", "fia_unpadding_idx"}},
  };
  return attnInTensorCandidates;
}

std::map<std::string, std::vector<std::string>>
GetAttnIntermediateTensorCandidates() {
  std::map<std::string, std::vector<std::string>>
      attnIntermediateTensorCandidates = {
          {"default",
           {"intermediate_q", "intermediate_k", "intermediate_v",
            "intermediate_self_attention"}},
          {"kv_quant_scale", {"intermediate_k_int8", "intermediate_v_int8"}},
          {"q_quant_scale", {"intermediate_q_int8"}},
          {"dequant_rope", {"intermediate_qkv_rope"}},
          {"fia", {"intermediate_q_bsnd", "intermediate_self_attention_bsnd"}}};
  return attnIntermediateTensorCandidates;
}

std::map<std::string, std::vector<std::string>>
GetAttnIntermediateTensorCandidatesPrefill() {
  std::map<std::string, std::vector<std::string>>
      attnIntermediateTensorCandidates = {
          {"default", {"intermediate_q", "intermediate_self_attention"}},
          {"kv_quant_scale", {"intermediate_k_int8", "intermediate_v_int8"}},
          {"q_quant_scale", {"intermediate_q_int8"}},
          {"dequant_rope", {"intermediate_qkv_rope"}},
          {"fia", {"intermediate_q_bsnd", "intermediate_self_attention_bsnd"}}};
  return attnIntermediateTensorCandidates;
}

template <typename NormParamType>
atb::Status ConstructAttentionQuantTensorMap(
    const FusionAttentionParam<NormParamType> &param,
    std::map<std::string, std::vector<std::string>> &attnInTensorCandidates,
    std::map<std::string, std::vector<std::string>>
        &attnIntermediateTensorCandidates,
    std::vector<std::string> &inTensorList,
    std::vector<std::string> &intermediateTensorList) {
  // translatedKV cache int8translatedTensor
  if (param.pageAttentionParam.quantType ==
      atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
    AddTensorToList(attnInTensorCandidates, "kv_quant_scale", inTensorList);
    if (!param.enableRopeQuantKvcache) {
      AddTensorToList(attnIntermediateTensorCandidates, "kv_quant_scale",
                      intermediateTensorList);
    }
    if (param.pageAttentionParam.hasQuantOffset) {
      AddTensorToList(attnInTensorCandidates, "kv_quant_offset", inTensorList);
    }
  }

  // translatedFA3translatedTensor
  if (!param.isPrefill &&
      param.pageAttentionParam.quantType ==
          atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
    AddTensorToList(attnIntermediateTensorCandidates, "q_quant_scale",
                    intermediateTensorList);
  }
  if (param.pageAttentionParam.quantType ==
      atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
    AddTensorToList(attnInTensorCandidates, "fa3_quant", inTensorList);
    AddTensorToList(attnIntermediateTensorCandidates, "kv_quant_scale",
                    intermediateTensorList);
  }
  return atb::NO_ERROR;
}

template <typename NormParamType>
std::map<std::string, uint32_t>
ConstructTensorMap(const FusionAttentionParam<NormParamType> &param,
                   uint32_t &inTensorNum, uint32_t &outTensorNum,
                   uint32_t &internalTensorNum) {
  auto attnInTensorCandidates = GetAttnInTensorCandidates();
  std::map<std::string, std::vector<std::string>>
      attnIntermediateTensorCandidates;
  if (param.isPrefill && param.enableXattention &&
      !param.isOneRecCrossAttention) {
    attnIntermediateTensorCandidates =
        GetAttnIntermediateTensorCandidatesPrefill();
  } else {
    attnIntermediateTensorCandidates = GetAttnIntermediateTensorCandidates();
  }

  std::vector<std::string> inTensorList = {};
  std::vector<std::string> intermediateTensorList = {};
  std::vector<std::string> outTensorList = {"out"};

  // translatedTensor
  AddTensorToList(attnInTensorCandidates, "default", inTensorList);
  // translatedAddRmsNormQuanttranslatedTensor
  if (param.enableAddNorm) {
    AddTensorToList(attnInTensorCandidates, "add_rmsnorm_quant", inTensorList);
  }
  if (param.isPrefill && param.isFIA &&
      param.aclnnFusedInferAttnParam.inputLayout == "BSND") {
    AddTensorToList(attnInTensorCandidates, "fia", inTensorList);
  }
  AddTensorToList(attnIntermediateTensorCandidates, "default",
                  intermediateTensorList);

  // translatedMask AlibitranslatedTensor
  if (param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
      param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
      param.selfAttentionParam.maskType ==
          atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
    AddTensorToList(attnInTensorCandidates, "alibi_mask_compress",
                    inTensorList);
  }

  // translatedTensor
  if (param.pageAttentionParam.compressType ==
      atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD) {
    AddTensorToList(attnInTensorCandidates, "compress_head_alibi",
                    inTensorList);
  } else if (param.pageAttentionParam.compressType ==
             atb::infer::PagedAttentionParam::CompressType::
                 COMPRESS_TYPE_KVHEAD_ROPE) {
    AddTensorToList(attnInTensorCandidates, "compress_head_rope", inTensorList);
  }
  // translatedTensor
  if (param.pageAttentionParam.calcType ==
          atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC ||
      param.isPrefixCacheWithoutChunk || param.isFIA) {
    AddTensorToList(attnInTensorCandidates, "speculate", inTensorList);
  }

  ConstructAttentionQuantTensorMap(param, attnInTensorCandidates,
                                   attnIntermediateTensorCandidates,
                                   inTensorList, intermediateTensorList);

  // translatedloratranslatedTensor
  if (param.supportLora) {
    if (param.useImMask) {
      AddTensorToList(attnInTensorCandidates, "lora_with_mask", inTensorList);
    }
    AddTensorToList(attnInTensorCandidates, "lora", inTensorList);
  }
  // translatedlccl all reduce int8translatedTensor
  if (param.selfOutLinearTensorParallelInfo.quantType !=
      atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
    AddTensorToList(attnInTensorCandidates, "reduce_quant", inTensorList);
  }

  // translatedlogN attentiontranslated
  if (param.pageAttentionParam.scaleType ==
      atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN) {
    AddTensorToList(attnInTensorCandidates, "log_n_scale", inTensorList);
  }

  // translated qk_norm translated Tensor
  if (param.useQKNorm) {
    AddTensorToList(attnInTensorCandidates, "qk_norm", inTensorList);
  }

  // translatedadd normtranslatedTensor
  if (param.enableAddNorm) {
    AddTensorToList(attnInTensorCandidates, "add_norm", inTensorList);
  }

  if (param.enableAddNorm) {
    outTensorList.push_back("out_add");
  }

  // translatedcmotranslatedTensor
  if (param.enablePreFetchWeight) {
    AddTensorToList(attnInTensorCandidates, "cmo_mlp_first_matmul_weight",
                    inTensorList);
  }

  // translated dequant rope tensor
  if (param.enableRopeQuantKvcache) {
    AddTensorToList(attnIntermediateTensorCandidates, "dequant_rope",
                    intermediateTensorList);
  }

  // Add flashcomm1.0 Tensor
  if (param.enableFlashComm) {
    AddTensorToList(attnInTensorCandidates, "flash_comm", inTensorList);
  }

  if (param.enableXattention) {
    AddTensorToList(attnInTensorCandidates, "x_attention", inTensorList);
  }

  if (!param.isPrefill && param.enableAclGraphPagedAttention) {
    AddTensorToList(attnInTensorCandidates, "acl_graph", inTensorList);
  }

  if (param.isPrefill && param.isFIA &&
      param.aclnnFusedInferAttnParam.inputLayout == "BSND") {
    AddTensorToList(attnIntermediateTensorCandidates, "fia",
                    intermediateTensorList);
  }

  inTensorNum = inTensorList.size();
  outTensorNum = outTensorList.size();
  internalTensorNum = intermediateTensorList.size();

  return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

template <typename NormParamType>
atb::Status
AddFAttnQKVLinearSplitNode(const FusionAttentionParam<NormParamType> &param,
                           atb::GraphParam &opGraph,
                           std::map<std::string, uint32_t> &tensorMap) {
  atb::Node qkvLinearSplitNode;
  CHECK_OPERATION_STATUS_RETURN(
      QKVLinearSplit(param, &qkvLinearSplitNode.operation));
  std::vector<std::string> qkvInTensorNames = {
      "in_input",           "in_norm_weight",    "in_norm_bias",
      "in_norm_new_weight", "in_norm_new_bias",  "in_weight_0",
      "in_scale_0",         "in_offset_0",       "in_descale_0",
      "in_bias_0",          "in_compress_idx_0", "in_weight_1",
      "in_scale_1",         "in_offset_1",       "in_descale_1",
      "in_bias_1",          "in_compress_idx_1", "in_weight_2",
      "in_scale_2",         "in_offset_2",       "in_descale_2",
      "in_bias_2",          "in_compress_idx_2",
  };
  // translatedAddRmsNormQuanttranslatedTensor
  if (param.enableAddNorm) {
    qkvInTensorNames.push_back("in_qkv_scale_fill");
    qkvInTensorNames.push_back("in_qkv_offset_fill");
    qkvInTensorNames.push_back("in_residual_add");
  }
  if (param.supportLora) {
    if (param.useImMask) {
      qkvInTensorNames.push_back("in_im_mask");
    }
    qkvInTensorNames.push_back("in_seq_len_cum_sum");
    qkvInTensorNames.push_back("in_lora_a_0");
    qkvInTensorNames.push_back("in_lora_b_0");
    qkvInTensorNames.push_back("in_lora_a_1");
    qkvInTensorNames.push_back("in_lora_b_1");
    qkvInTensorNames.push_back("in_lora_a_2");
    qkvInTensorNames.push_back("in_lora_b_2");
  }
  if (param.useQKNorm) {
    qkvInTensorNames.push_back("in_q_norm_weight");
    qkvInTensorNames.push_back("in_k_norm_weight");
  }
  if (param.enableFlashComm) {
    qkvInTensorNames.push_back("send_counts");
    qkvInTensorNames.push_back("sdispls");
    qkvInTensorNames.push_back("send_count");
    qkvInTensorNames.push_back("recv_counts");
    qkvInTensorNames.push_back("rdispls");
    qkvInTensorNames.push_back("recv_count");
    qkvInTensorNames.push_back("fake_ag_shape");
  }

  if (param.enableSplitRmsNormRope) {
    qkvInTensorNames.push_back("in_cos_embed");
    qkvInTensorNames.push_back("in_sin_embed");
  }

  qkvLinearSplitNode.inTensorIds =
      GetTensorIdxList(tensorMap, qkvInTensorNames);
  std::vector<std::string> qkvOutTensorNames;
  if (param.isPrefill && param.enableXattention &&
      !param.isOneRecCrossAttention) {
    qkvOutTensorNames = {"intermediate_q", "in_decode_k_cache",
                         "in_decode_v_cache"};
  } else {
    qkvOutTensorNames = {"intermediate_q", "intermediate_k", "intermediate_v"};
  }
  if (param.enableRopeQuantKvcache) { // 3 -> 1
    qkvOutTensorNames = {"intermediate_qkv_rope"};
  }
  if (param.enableAddNorm) {
    qkvOutTensorNames.push_back("out_add");
  }
  qkvLinearSplitNode.outTensorIds =
      GetTensorIdxList(tensorMap, qkvOutTensorNames);
  opGraph.nodes.push_back(qkvLinearSplitNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t
AddRopeQuantKvcacheOperation(atb::GraphParam &opGraph,
                             const FusionAttentionParam<NormParamType> &param,
                             std::map<std::string, uint32_t> &tensorMap) {
  atb::Node dequantRopeQuantKvcacheNode;
  AclNNDequantRopeQuantKvcacheParam aclnnParam;

  int64_t sizeSpiltsZero =
      CheckIntMulOverFlow(param.selfAttentionParam.headNum, param.headDim);
  int64_t sizeSpiltsOne =
      CheckIntMulOverFlow(param.selfAttentionParam.kvHeadNum, param.headDim);
  aclnnParam.sizeSpilts = {sizeSpiltsZero, sizeSpiltsOne, sizeSpiltsOne};

  aclnnParam.kvOutput = true;
  aclnnParam.quantMode = "static";
  aclnnParam.layout = "BSND";
  LinearQuantType quantType = GetLinearQuantType(
      param.packQuantType, param.layerLinearQuantType[Q_LINEAR_INDEX],
      param.enableNormQuantOp);
  aclnnParam.enableDequant =
      (!param.isPrefill && param.isBF16 &&
       (quantType == LINEAR_W8A8_QUANT || quantType == LINEAR_W8A8_DEQUANT));
  dequantRopeQuantKvcacheNode.operation =
      new atb_speed::common::DequantRopeQuantKvcacheOperation(
          "aclnnDequantRopeQuantKvcacheNode", aclnnParam);
  dequantRopeQuantKvcacheNode.inTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_qkv_rope"),        // 1: input_x
      GetTensorIdx(tensorMap, "in_cos_embed"),                 // 2: cos
      GetTensorIdx(tensorMap, "in_sin_embed"),                 // 3: sin
      GetTensorIdx(tensorMap, "in_k_cache"),                   // 4: k_cache
      GetTensorIdx(tensorMap, "in_v_cache"),                   // 5: v_cache
      GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"), // 6: indices
      GetTensorIdx(tensorMap, "in_k_quant_scale"),             // 7: scale_k
      GetTensorIdx(tensorMap, "in_v_quant_scale"),             // 8: scale_v
      GetTensorIdx(tensorMap, "in_k_quant_offset"),            // 9: offset_k
      GetTensorIdx(tensorMap, "in_v_quant_offset"),            // 10: offset_v
  };
  if (aclnnParam.enableDequant) {
    dequantRopeQuantKvcacheNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_descale_0")); // 11: weight_scale
    dequantRopeQuantKvcacheNode.inTensorIds.push_back(
        GetTensorIdx(tensorMap, "in_bias_0")); // 12: bias
  }
  if (param.isPrefill && param.enableXattention &&
      !param.isOneRecCrossAttention) {
    dequantRopeQuantKvcacheNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),    // q_out // 1024, 8, 128
        GetTensorIdx(tensorMap, "in_decode_k_cache"), // k_out // 1024, 1, 128
        GetTensorIdx(tensorMap, "in_decode_v_cache"), // v_out // 1024, 1, 128
    };
  } else {
    dequantRopeQuantKvcacheNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"), // q_out // 1024, 8, 128
        GetTensorIdx(tensorMap, "intermediate_k"), // k_out // 1024, 1, 128
        GetTensorIdx(tensorMap, "intermediate_v"), // v_out // 1024, 1, 128
    };
  }
  opGraph.nodes.push_back(dequantRopeQuantKvcacheNode);

  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddFAttnRopeNode(const FusionAttentionParam<NormParamType> &param,
                             atb::GraphParam &opGraph,
                             std::map<std::string, uint32_t> &tensorMap) {
  atb::Node ropeNode;
  atb_speed::common::RotaryPositionEmbeddingParam ropeParam;
  ropeParam.rotaryType = param.rotaryType;
  ropeParam.isFA = param.isFA;
  ropeParam.headDim = param.headDim;
  ropeParam.headNum = param.selfAttentionParam.headNum;
  ropeParam.kvHeadNum = param.selfAttentionParam.kvHeadNum;
  ropeParam.ropeParam = param.ropeParam;

  RotaryPositionEmbedding(ropeParam, &ropeNode.operation);
  if (param.isPrefill && param.enableXattention &&
      !param.isOneRecCrossAttention) {
    ropeNode.inTensorIds = {// [B,S,N,D] PA [BS,ND]
                            GetTensorIdx(tensorMap, "intermediate_q"),
                            GetTensorIdx(tensorMap, "in_decode_k_cache"),
                            GetTensorIdx(tensorMap, "in_cos_embed"),
                            GetTensorIdx(tensorMap, "in_sin_embed"),
                            GetTensorIdx(tensorMap, "in_seq_len")};
  } else {
    ropeNode.inTensorIds = {// [B,S,N,D] PA [BS,ND]
                            GetTensorIdx(tensorMap, "intermediate_q"),
                            GetTensorIdx(tensorMap, "intermediate_k"),
                            GetTensorIdx(tensorMap, "in_cos_embed"),
                            GetTensorIdx(tensorMap, "in_sin_embed"),
                            GetTensorIdx(tensorMap, "in_seq_len")};
  }

  if (!param.isFA) {
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(0) = &SqueezeHeadNumHeadDim;
    ropeNode.inTensorReshapeFuncs.at(1) = &SqueezeHeadNumHeadDim;
  }
  if (param.isPrefill && param.enableXattention &&
      !param.isOneRecCrossAttention) {
    ropeNode.outTensorIds = {
        // FA [B,S,N,D] PA [BS,N,D]
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "in_decode_k_cache"),
    };
  } else {
    ropeNode.outTensorIds = {
        // FA [B,S,N,D] PA [BS,N,D]
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "intermediate_k"),
    };
  }
  opGraph.nodes.push_back(ropeNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status
AddKVValueQuantNode(const FusionAttentionParam<NormParamType> &param,
                    atb::GraphParam &opGraph,
                    std::map<std::string, uint32_t> &tensorMap, bool isK) {
  atb::Node kvValueQuantNode;
  atb::infer::ElewiseParam kvValueQuantParam;
  kvValueQuantParam.elewiseType =
      atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
  CREATE_OPERATION(kvValueQuantParam, &kvValueQuantNode.operation);
  if (isK) {
    if (param.isPrefill && param.enableXattention &&
      !param.isOneRecCrossAttention) {
      kvValueQuantNode.inTensorIds = {
          GetTensorIdx(tensorMap, "in_decode_k_cache"),
          GetTensorIdx(tensorMap, "in_k_quant_scale"),
          GetTensorIdx(tensorMap, "in_k_quant_offset"),
      };
    } else {
      kvValueQuantNode.inTensorIds = {
          GetTensorIdx(tensorMap, "intermediate_k"),
          GetTensorIdx(tensorMap, "in_k_quant_scale"),
          GetTensorIdx(tensorMap, "in_k_quant_offset"),
      };
    }
    kvValueQuantNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_k_int8")};
  } else {
    if (param.isPrefill && param.enableXattention &&
      !param.isOneRecCrossAttention) {
      kvValueQuantNode.inTensorIds = {
          GetTensorIdx(tensorMap, "in_decode_v_cache"),
          GetTensorIdx(tensorMap, "in_v_quant_scale"),
          GetTensorIdx(tensorMap, "in_v_quant_offset"),
      };
    } else {
      kvValueQuantNode.inTensorIds = {
          GetTensorIdx(tensorMap, "intermediate_v"),
          GetTensorIdx(tensorMap, "in_v_quant_scale"),
          GetTensorIdx(tensorMap, "in_v_quant_offset"),
      };
    }
    kvValueQuantNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_v_int8")};
  }
  kvValueQuantNode.inTensorReshapeFuncs.resize(
      kvValueQuantNode.inTensorIds.size());
  kvValueQuantNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape,
                                                 atb::Dims &newShape) {
    UnsqueezeHeadNumHeadDim(oldShape, newShape,
                            param.selfAttentionParam.kvHeadNum, param.headDim);
  };
  kvValueQuantNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape,
                                                 atb::Dims &newShape) {
    UnsqueezeHeadNumHeadDim(oldShape, newShape,
                            param.selfAttentionParam.kvHeadNum, param.headDim);
  };
  opGraph.nodes.push_back(kvValueQuantNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddQKVQuantNode(const FusionAttentionParam<NormParamType> &param,
                            atb::GraphParam &opGraph,
                            std::map<std::string, uint32_t> &tensorMap,
                            std::string nodeType) {
  atb::Node qkvQuantNode;
  atb::infer::ElewiseParam qkvQuantParam;
  qkvQuantParam.elewiseType =
      atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
  CREATE_OPERATION(qkvQuantParam, &qkvQuantNode.operation);
  if (nodeType == "Q") {
    qkvQuantNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_q"),
                                GetTensorIdx(tensorMap, "in_q_quant_scale"),
                                GetTensorIdx(tensorMap, "q_offset")};
    qkvQuantNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q_int8")};
  } else if (nodeType == "K") {
    if (param.isPrefill && param.enableXattention &&
      !param.isOneRecCrossAttention) {
      qkvQuantNode.inTensorIds = {GetTensorIdx(tensorMap, "in_decode_k_cache"),
                                  GetTensorIdx(tensorMap, "in_k_quant_scale"),
                                  GetTensorIdx(tensorMap, "kv_offset")};
    } else {
      qkvQuantNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_k"),
                                  GetTensorIdx(tensorMap, "in_k_quant_scale"),
                                  GetTensorIdx(tensorMap, "kv_offset")};
    }
    qkvQuantNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_k_int8")};
  } else if (nodeType == "V") {
    if (param.isPrefill && param.enableXattention &&
      !param.isOneRecCrossAttention) {
      qkvQuantNode.inTensorIds = {
          GetTensorIdx(tensorMap, "in_decode_v_cache"),
          GetTensorIdx(tensorMap, "in_v_quant_scale"),
          GetTensorIdx(tensorMap, "kv_offset"),
      };
    } else {
      qkvQuantNode.inTensorIds = {
          GetTensorIdx(tensorMap, "intermediate_v"),
          GetTensorIdx(tensorMap, "in_v_quant_scale"),
          GetTensorIdx(tensorMap, "kv_offset"),
      };
    }
    qkvQuantNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_v_int8")};
  }
  qkvQuantNode.inTensorReshapeFuncs.resize(qkvQuantNode.inTensorIds.size());
  opGraph.nodes.push_back(qkvQuantNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddPadNode(atb::GraphParam &opGraph,
                       const FusionAttentionParam<NormParamType> & /*param*/,
                       std::map<std::string, uint32_t> &tensorMap) {
  atb::infer::GatherParam padParam;
  atb::Node padqNode;
  CHECK_OPERATION_STATUS_RETURN(
      atb::CreateOperation(padParam, &padqNode.operation));
  padqNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_q", "fia_padding_idx"});
  padqNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_q_bsnd"});
  opGraph.nodes.push_back(padqNode);

  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddUnpadNode(atb::GraphParam &opGraph,
                         const FusionAttentionParam<NormParamType> & /*param*/,
                         std::map<std::string, uint32_t> &tensorMap) {
  atb::infer::GatherParam unpadParam;
  atb::Node unpadNode;
  CHECK_OPERATION_STATUS_RETURN(
      atb::CreateOperation(unpadParam, &unpadNode.operation));
  unpadNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_self_attention_bsnd", "fia_unpadding_idx"});
  unpadNode.inTensorReshapeFuncs.resize(unpadNode.inTensorIds.size());
  unpadNode.inTensorReshapeFuncs.at(0) = &SqueezeBatchAndHiddenSize;
  unpadNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_self_attention"});
  opGraph.nodes.push_back(unpadNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status
AddSelfOutLinearParallelNode(const FusionAttentionParam<NormParamType> &param,
                             atb::GraphParam &opGraph,
                             std::map<std::string, uint32_t> &tensorMap) {
  atb::Node selfOutLinearParallelNode;
  atb_speed::common::LinearParallelParam selfOutLinearParam;
  if (param.enableFlashComm) {
    selfOutLinearParam.parallelType = atb_speed::common::REDUCE_SCATTER;
  } else {
    selfOutLinearParam.parallelType = atb_speed::common::ROW_PARALLEL;
  }
  selfOutLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
      param.denseQuantType ==
              atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
          ? param.packQuantType
          : param.denseQuantType,
      param.layerLinearQuantType[DENSE_LINEAR_INDEX], false,
      param.layerLinearDescs[DENSE_LINEAR_INDEX]);
  selfOutLinearParam.biasAfterSync =
      param.selfOutLinearTensorParallelInfo.worldSize > 1 &&
      selfOutLinearParam.fusionLinearParam.quantType ==
          atb_speed::common::LinearQuantType::NO_QUANT &&
      param.selfAttnHasBias;
  selfOutLinearParam.fusionLinearParam.isBF16 = param.isBF16;
  selfOutLinearParam.fusionLinearParam.hasBias =
      param.selfAttnHasBias && !selfOutLinearParam.biasAfterSync;
  selfOutLinearParam.fusionLinearParam.supportLora = param.supportLora;
  selfOutLinearParam.fusionLinearParam.useImMask = param.useImMask;
  selfOutLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
  selfOutLinearParam.fusionLinearParam.transposeType =
      param.layerLinearTransposeType[DENSE_LINEAR_INDEX];
  selfOutLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
  selfOutLinearParam.fusionLinearParam.matmulBackend = param.matmulBackend;
  selfOutLinearParam.fusionLinearParam.isPrefill = param.isPrefill;
  selfOutLinearParam.tensorParallelInfo = param.selfOutLinearTensorParallelInfo;
  selfOutLinearParam.supportLcoc = param.supportLcoc;
  selfOutLinearParam.enableMC2 = param.enableMC2;
  CHECK_OPERATION_STATUS_RETURN(
      LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation));
  std::vector<std::string> denseInTensorNames = {"intermediate_self_attention",
                                                 "in_weight_dense",
                                                 "in_scale_dense",
                                                 "in_offset_dense",
                                                 "in_descale_dense",
                                                 "in_bias_dense",
                                                 "in_compress_idx_dense"};
  if (param.supportLora) {
    if (param.useImMask) {
      denseInTensorNames.push_back("in_im_mask");
    }
    denseInTensorNames.push_back("in_seq_len_cum_sum");
    denseInTensorNames.push_back("in_dense_lora_a");
    denseInTensorNames.push_back("in_dense_lora_b");
  }
  if (param.selfOutLinearTensorParallelInfo.quantType !=
      atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
    denseInTensorNames.push_back("in_reduce_quant_scale");
    denseInTensorNames.push_back("in_reduce_quant_offset");
    denseInTensorNames.push_back("in_gather_quant_scale");
    denseInTensorNames.push_back("in_gather_quant_offset");
  }
  if (selfOutLinearParam.parallelType == atb_speed::common::REDUCE_SCATTER) {
    denseInTensorNames.push_back("send_counts");
    denseInTensorNames.push_back("sdispls");
    denseInTensorNames.push_back("recv_count");
    denseInTensorNames.push_back("fake_rs_shape");
  }
  selfOutLinearParallelNode.inTensorIds =
      GetTensorIdxList(tensorMap, denseInTensorNames);
  if (param.isFIA) {
    selfOutLinearParallelNode.inTensorReshapeFuncs.resize(
        selfOutLinearParallelNode.inTensorIds.size());
    selfOutLinearParallelNode.inTensorReshapeFuncs.at(0) =
        param.aclnnFusedInferAttnParam.inputLayout == "BSND"
            ? &SqueezeBatchAndHiddenSize
            : &SqueezeHeadNumHeadDim;
  } else if (!param.isFA) {
    selfOutLinearParallelNode.inTensorReshapeFuncs.resize(
        selfOutLinearParallelNode.inTensorIds.size());
    selfOutLinearParallelNode.inTensorReshapeFuncs.at(0) =
        &SqueezeHeadNumHeadDim;
  }
  selfOutLinearParallelNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
  opGraph.nodes.push_back(selfOutLinearParallelNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status
AddCrossOutLinearParallelNode(const FusionAttentionParam<NormParamType> &param,
                              atb::GraphParam &opGraph,
                              std::map<std::string, uint32_t> &tensorMap) {
  atb::Node selfOutLinearParallelNode;
  atb_speed::common::LinearParallelParam selfOutLinearParam;
  selfOutLinearParam.parallelType = atb_speed::common::ROW_PARALLEL;
  selfOutLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
      param.denseQuantType ==
              atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
          ? param.packQuantType
          : param.denseQuantType,
      param.layerLinearQuantType[DENSE_LINEAR_INDEX], false,
      param.layerLinearDescs[DENSE_LINEAR_INDEX]);
  selfOutLinearParam.biasAfterSync =
      param.selfOutLinearTensorParallelInfo.worldSize > 1 &&
      selfOutLinearParam.fusionLinearParam.quantType ==
          atb_speed::common::LinearQuantType::NO_QUANT &&
      param.selfAttnHasBias;
  selfOutLinearParam.fusionLinearParam.isBF16 = param.isBF16;
  selfOutLinearParam.fusionLinearParam.hasBias =
      param.selfAttnHasBias && !selfOutLinearParam.biasAfterSync;
  selfOutLinearParam.fusionLinearParam.supportLora = param.supportLora;
  selfOutLinearParam.fusionLinearParam.useImMask = param.useImMask;
  selfOutLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
  selfOutLinearParam.fusionLinearParam.transposeType =
      param.layerLinearTransposeType[DENSE_LINEAR_INDEX];
  selfOutLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
  selfOutLinearParam.fusionLinearParam.matmulBackend = param.matmulBackend;
  selfOutLinearParam.fusionLinearParam.isPrefill = param.isPrefill;
  selfOutLinearParam.tensorParallelInfo = param.selfOutLinearTensorParallelInfo;
  selfOutLinearParam.supportLcoc = param.supportLcoc;
  selfOutLinearParam.enableMC2 = param.enableMC2;
  CHECK_OPERATION_STATUS_RETURN(
      LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation));
  std::vector<std::string> denseInTensorNames = {
      "intermediate_self_attention", "in_cross_dense_weight",
      "in_cross_dense_bias",         "in_cross_dense_descale",
      "in_cross_dense_offset",       "in_cross_dense_scale",
      "in_cross_dense_compress_idx",
  };
  if (param.supportLora) {
    if (param.useImMask) {
      denseInTensorNames.push_back("in_im_mask");
    }
    denseInTensorNames.push_back("in_seq_len_cum_sum");
    denseInTensorNames.push_back("in_dense_lora_a");
    denseInTensorNames.push_back("in_dense_lora_b");
  }
  if (param.selfOutLinearTensorParallelInfo.quantType !=
      atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
    denseInTensorNames.push_back("in_reduce_quant_scale");
    denseInTensorNames.push_back("in_reduce_quant_offset");
    denseInTensorNames.push_back("in_gather_quant_scale");
    denseInTensorNames.push_back("in_gather_quant_offset");
  }
  selfOutLinearParallelNode.inTensorIds =
      GetTensorIdxList(tensorMap, denseInTensorNames);
  selfOutLinearParallelNode.inTensorReshapeFuncs.resize(
      selfOutLinearParallelNode.inTensorIds.size());
  selfOutLinearParallelNode.inTensorReshapeFuncs[0] =
      [&](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2] * oldShape.dims[3];
      };
  selfOutLinearParallelNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
  opGraph.nodes.push_back(selfOutLinearParallelNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status
AddCrossAttnQKVProjectionNodes(const FusionAttentionParam<NormParamType> &param,
                               atb::GraphParam &opGraph,
                               std::map<std::string, uint32_t> &tensorMap) {
  const bool stop_with_identity_k =
      DebugStopOneRecCrossLayer0WithIdentityK(param);
  const bool stop_with_identity_v =
      DebugStopOneRecCrossLayer0WithIdentityV(param);
  const bool stop_with_identity_kv =
      DebugStopOneRecCrossLayer0WithIdentityKv(param);
  const bool stop_after_k_only = DebugStopOneRecCrossLayer0AfterKOnly(param);
  const bool stop_after_kv_only = DebugStopOneRecCrossLayer0AfterKvOnly(param);
  if (param.isPrefill) {
    if (!stop_after_kv_only && !stop_after_k_only && !stop_with_identity_k &&
        !stop_with_identity_v && !stop_with_identity_kv) {
      atb::Node qProjectionNode;
      atb_speed::common::NormLinearParam<NormParamType> qNormLinearParam;
      qNormLinearParam.fusionLinearParam.quantType =
          atb_speed::common::GetLinearQuantType(
              param.packQuantType, param.layerLinearQuantType[Q_LINEAR_INDEX],
              param.enableNormQuantOp, param.layerLinearDescs[Q_LINEAR_INDEX]);
      qNormLinearParam.fusionLinearParam.hasBias = param.qkvHasBias;
      qNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
      qNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
      qNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
      qNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
      qNormLinearParam.fusionLinearParam.transposeType =
          param.layerLinearTransposeType[Q_LINEAR_INDEX];
      qNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
      qNormLinearParam.fusionLinearParam.isPrefill = param.isPrefill;
      qNormLinearParam.fusionLinearParam.matmulBackend = param.matmulBackend;
      qNormLinearParam.skipNorm = param.skipNorm;
      qNormLinearParam.normHasBias = param.normHasBias;
      qNormLinearParam.normParamType = param.normParamType;
      qNormLinearParam.normQuantParamType = param.normQuantParamType;

      if (param.enableAddNorm && !param.skipNorm && !param.supportLora) {
        atb::GraphParam qOpGraph;
        qOpGraph.name = "CrossAttnQAddRmsNormLinear";
        std::vector<std::string> qInTensorList = {
            "in_input",         "in_residual_add", "in_norm_weight",
            "in_linear_weight", "in_scale",        "in_offset",
            "in_descale",       "in_bias",         "in_compress_idx",
        };
        std::vector<std::string> qOutTensorList = {"out_q", "out_add"};
        std::vector<std::string> qIntermediateTensorList = {
            "intermediate_norm",
            "intermediate_rstd",
        };
        qOpGraph.inTensorNum = qInTensorList.size();
        qOpGraph.outTensorNum = qOutTensorList.size();
        qOpGraph.internalTensorNum = qIntermediateTensorList.size();
        std::map<std::string, uint32_t> qTensorMap =
            GetTensorMap(qInTensorList, qOutTensorList, qIntermediateTensorList);

        atb::Node addRmsNormNode;
        addRmsNormNode.operation = new atb_speed::common::AddRmsNormOperation(
            "AclnnCrossAttnQAddRmsNormNode",
            param.normParamType.normParam.epsilon);
        addRmsNormNode.inTensorIds = GetTensorIdxList(
            qTensorMap, {"in_residual_add", "in_input", "in_norm_weight"});
        addRmsNormNode.outTensorIds = GetTensorIdxList(
            qTensorMap, {"intermediate_norm", "intermediate_rstd", "out_add"});
        qOpGraph.nodes.push_back(addRmsNormNode);

        atb::Node linearNode;
        atb_speed::common::FusionLinearParam linearParam =
            qNormLinearParam.fusionLinearParam;
        CHECK_OPERATION_STATUS_RETURN(
            FusionLinear(linearParam, &linearNode.operation));
        linearNode.inTensorIds = GetTensorIdxList(
            qTensorMap,
            {"intermediate_norm", "in_linear_weight", "in_scale", "in_offset",
             "in_descale", "in_bias", "in_compress_idx"});
        linearNode.outTensorIds = {GetTensorIdx(qTensorMap, "out_q")};
        qOpGraph.nodes.push_back(linearNode);

        CHECK_OPERATION_STATUS_RETURN(
            atb::CreateOperation(qOpGraph, &qProjectionNode.operation));

        std::vector<std::string> qInTensor = {
            "intermediate_self_residual_out",
            "in_residual_add",
            "in_cross_attn_norm_weight",
            "in_cross_weight_0",
            "in_cross_scale_0",
            "in_cross_offset_0",
            "in_cross_descale_0",
            "in_cross_bias_0",
            "in_cross_compress_idx_0",
        };
        qProjectionNode.inTensorIds = GetTensorIdxList(tensorMap, qInTensor);
        qProjectionNode.outTensorIds =
            GetTensorIdxList(tensorMap, {"intermediate_q", "out_add"});
      } else {
        CHECK_OPERATION_STATUS_RETURN(
            atb_speed::common::NormLinear<NormParamType>(
                qNormLinearParam, &qProjectionNode.operation));
        std::vector<std::string> qInTensor = {
            "intermediate_self_residual_out",
            "in_cross_attn_norm_weight",
            "in_cross_attn_norm_bias",
            "in_cross_attn_norm_new_weight",
            "in_cross_attn_norm_new_bias",
            "in_cross_weight_0",
            "in_cross_bias_0",
            "in_cross_descale_0",
            "in_cross_offset_0",
            "in_cross_scale_0",
            "in_cross_compress_idx_0",
        };
        if (param.supportLora) {
          if (param.useImMask) {
            qInTensor.push_back("in_im_mask");
          }
          qInTensor.push_back("in_seq_len_cum_sum");
          qInTensor.push_back("in_lora_a_0");
          qInTensor.push_back("in_lora_b_0");
        }
        qProjectionNode.inTensorIds = GetTensorIdxList(tensorMap, qInTensor);
        qProjectionNode.outTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_q")};
      }
      opGraph.nodes.push_back(qProjectionNode);
    }

    if (!stop_with_identity_k && !stop_with_identity_kv) {
      atb::Node kProjectionNode;
      atb_speed::common::NormLinearParam<NormParamType> kNormLinearParam;
      kNormLinearParam.fusionLinearParam.quantType =
          atb_speed::common::GetLinearQuantType(
              param.packQuantType, param.layerLinearQuantType[K_LINEAR_INDEX],
              param.enableNormQuantOp, param.layerLinearDescs[K_LINEAR_INDEX]);
      kNormLinearParam.fusionLinearParam.hasBias = param.qkvHasBias;
      kNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
      kNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
      kNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
      kNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
      kNormLinearParam.fusionLinearParam.transposeType = 1;
      kNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
      kNormLinearParam.fusionLinearParam.isPrefill = param.isPrefill;
      kNormLinearParam.fusionLinearParam.matmulBackend = param.matmulBackend;
      kNormLinearParam.skipNorm = true;
      kNormLinearParam.normHasBias = param.normHasBias;
      kNormLinearParam.normParamType = param.normParamType;
      kNormLinearParam.normQuantParamType = param.normQuantParamType;
      CHECK_OPERATION_STATUS_RETURN(atb_speed::common::NormLinear<NormParamType>(
          kNormLinearParam, &kProjectionNode.operation));
      std::vector<std::string> kInTensor = {
          "in_encoder_output",
          "in_cross_attn_norm_weight",
          "in_cross_attn_norm_bias",
          "in_cross_attn_norm_new_weight",
          "in_cross_attn_norm_new_bias",
          "in_cross_weight_1",
          "in_cross_scale_1",
          "in_cross_offset_1",
          "in_cross_descale_1",
          "in_cross_bias_1",
          "in_cross_compress_idx_1",
      };
      if (param.supportLora) {
        if (param.useImMask) {
          kInTensor.push_back("in_im_mask");
        }
        kInTensor.push_back("in_seq_len_cum_sum");
        kInTensor.push_back("in_lora_a_1");
        kInTensor.push_back("in_lora_b_1");
      }
      kProjectionNode.inTensorIds = GetTensorIdxList(tensorMap, kInTensor);
      kProjectionNode.outTensorIds = {
          GetTensorIdx(tensorMap, "in_cross_k_cache")};
      opGraph.nodes.push_back(kProjectionNode);
      if (stop_after_k_only) {
        return atb::NO_ERROR;
      }
    }

    if (!stop_with_identity_v && !stop_with_identity_kv) {
      atb::Node vProjectionNode;
      atb_speed::common::NormLinearParam<NormParamType> vNormLinearParam;
      vNormLinearParam.fusionLinearParam.quantType =
          atb_speed::common::GetLinearQuantType(
              param.packQuantType, param.layerLinearQuantType[V_LINEAR_INDEX],
              param.enableNormQuantOp, param.layerLinearDescs[V_LINEAR_INDEX]);
      vNormLinearParam.fusionLinearParam.hasBias = param.qkvHasBias;
      vNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
      vNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
      vNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
      vNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
      vNormLinearParam.fusionLinearParam.transposeType = 1;
      vNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
      vNormLinearParam.fusionLinearParam.isPrefill = param.isPrefill;
      vNormLinearParam.fusionLinearParam.matmulBackend = param.matmulBackend;
      vNormLinearParam.skipNorm = true;
      vNormLinearParam.normHasBias = param.normHasBias;
      vNormLinearParam.normParamType = param.normParamType;
      vNormLinearParam.normQuantParamType = param.normQuantParamType;
      CHECK_OPERATION_STATUS_RETURN(
          atb_speed::common::NormLinear<NormParamType>(
              vNormLinearParam, &vProjectionNode.operation));
      std::vector<std::string> vInTensor = {
          "in_encoder_output",
          "in_cross_attn_norm_weight",
          "in_cross_attn_norm_bias",
          "in_cross_attn_norm_new_weight",
          "in_cross_attn_norm_new_bias",
          "in_cross_weight_2",
          "in_cross_scale_2",
          "in_cross_offset_2",
          "in_cross_descale_2",
          "in_cross_bias_2",
          "in_cross_compress_idx_2",
      };
      if (param.supportLora) {
        if (param.useImMask) {
          vInTensor.push_back("in_im_mask");
        }
        vInTensor.push_back("in_seq_len_cum_sum");
        vInTensor.push_back("in_lora_a_2");
        vInTensor.push_back("in_lora_b_2");
      }
      vProjectionNode.inTensorIds = GetTensorIdxList(tensorMap, vInTensor);
      vProjectionNode.outTensorIds = {
          GetTensorIdx(tensorMap, "in_cross_v_cache")};
      opGraph.nodes.push_back(vProjectionNode);
    }
  } else {
    atb::Node qProjectionNode;
    atb_speed::common::NormLinearParam<NormParamType> qNormLinearParam;
    qNormLinearParam.fusionLinearParam.quantType =
        atb_speed::common::GetLinearQuantType(
            param.packQuantType, param.layerLinearQuantType[Q_LINEAR_INDEX],
            param.enableNormQuantOp, param.layerLinearDescs[Q_LINEAR_INDEX]);
    qNormLinearParam.fusionLinearParam.hasBias = param.qkvHasBias;
    qNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    qNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
    qNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
    qNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    qNormLinearParam.fusionLinearParam.transposeType =
        param.layerLinearTransposeType[Q_LINEAR_INDEX];
    qNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    qNormLinearParam.fusionLinearParam.isPrefill = param.isPrefill;
    qNormLinearParam.fusionLinearParam.matmulBackend = param.matmulBackend;
    qNormLinearParam.skipNorm = param.skipNorm;
    qNormLinearParam.normHasBias = param.normHasBias;
    qNormLinearParam.normParamType = param.normParamType;
    qNormLinearParam.normQuantParamType = param.normQuantParamType;
    if (param.enableAddNorm && !param.skipNorm && !param.supportLora) {
      atb::GraphParam qOpGraph;
      qOpGraph.name = "CrossAttnQAddRmsNormLinear";
      std::vector<std::string> qInTensorList = {
          "in_input",         "in_residual_add", "in_norm_weight",
          "in_linear_weight", "in_scale",        "in_offset",
          "in_descale",       "in_bias",         "in_compress_idx",
      };
      std::vector<std::string> qOutTensorList = {"out_q", "out_add"};
      std::vector<std::string> qIntermediateTensorList = {
          "intermediate_norm",
          "intermediate_rstd",
      };
      qOpGraph.inTensorNum = qInTensorList.size();
      qOpGraph.outTensorNum = qOutTensorList.size();
      qOpGraph.internalTensorNum = qIntermediateTensorList.size();
      std::map<std::string, uint32_t> qTensorMap =
          GetTensorMap(qInTensorList, qOutTensorList, qIntermediateTensorList);

      atb::Node addRmsNormNode;
      addRmsNormNode.operation = new atb_speed::common::AddRmsNormOperation(
          "AclnnCrossAttnQAddRmsNormNode",
          param.normParamType.normParam.epsilon);
      addRmsNormNode.inTensorIds = GetTensorIdxList(
          qTensorMap, {"in_residual_add", "in_input", "in_norm_weight"});
      addRmsNormNode.outTensorIds = GetTensorIdxList(
          qTensorMap, {"intermediate_norm", "intermediate_rstd", "out_add"});
      qOpGraph.nodes.push_back(addRmsNormNode);

      atb::Node linearNode;
      atb_speed::common::FusionLinearParam linearParam =
          qNormLinearParam.fusionLinearParam;
      CHECK_OPERATION_STATUS_RETURN(
          FusionLinear(linearParam, &linearNode.operation));
      linearNode.inTensorIds =
          GetTensorIdxList(qTensorMap, {"intermediate_norm", "in_linear_weight",
                                        "in_scale", "in_offset", "in_descale",
                                        "in_bias", "in_compress_idx"});
      linearNode.outTensorIds = {GetTensorIdx(qTensorMap, "out_q")};
      qOpGraph.nodes.push_back(linearNode);

      CHECK_OPERATION_STATUS_RETURN(
          atb::CreateOperation(qOpGraph, &qProjectionNode.operation));

      std::vector<std::string> qInTensor = {
          "intermediate_self_residual_out",
          "in_residual_add",
          "in_cross_attn_norm_weight",
          "in_cross_weight_0",
          "in_cross_scale_0",
          "in_cross_offset_0",
          "in_cross_descale_0",
          "in_cross_bias_0",
          "in_cross_compress_idx_0",
      };
      qProjectionNode.inTensorIds = GetTensorIdxList(tensorMap, qInTensor);
      qProjectionNode.outTensorIds =
          GetTensorIdxList(tensorMap, {"intermediate_q", "out_add"});
    } else {
      CHECK_OPERATION_STATUS_RETURN(
          atb_speed::common::NormLinear<NormParamType>(
              qNormLinearParam, &qProjectionNode.operation));

      std::vector<std::string> qInTensor = {
          "intermediate_self_residual_out",
          "in_cross_attn_norm_weight",
          "in_cross_attn_norm_bias",
          "in_cross_attn_norm_new_weight",
          "in_cross_attn_norm_new_bias",
          "in_cross_weight_0",
          "in_cross_scale_0",
          "in_cross_offset_0",
          "in_cross_descale_0",
          "in_cross_bias_0",
          "in_cross_compress_idx_0",
      };
      if (param.supportLora) {
        if (param.useImMask) {
          qInTensor.push_back("in_im_mask");
        }
        qInTensor.push_back("in_seq_len_cum_sum");
        qInTensor.push_back("in_lora_a_0");
        qInTensor.push_back("in_lora_b_0");
      }
      qProjectionNode.inTensorIds = GetTensorIdxList(tensorMap, qInTensor);
      qProjectionNode.outTensorIds = {
          GetTensorIdx(tensorMap, "intermediate_q")};
    }
    opGraph.nodes.push_back(qProjectionNode);
  }

  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status
AddCrossAttnCacheUpdateNode(const FusionAttentionParam<NormParamType> &param,
                            atb::GraphParam &opGraph,
                            std::map<std::string, uint32_t> &tensorMap) {
  atb::Node reshapeAndCacheNode;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(
      param.reshapeCacheParm, &reshapeAndCacheNode.operation));
  reshapeAndCacheNode.inTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_k"),
      GetTensorIdx(tensorMap, "intermediate_v"),
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
      GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
  };
  reshapeAndCacheNode.inTensorReshapeFuncs.resize(2);
  auto reshapeKVFunc = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
    size_t dim = 0;
    newShape.dims[dim++] = oldShape.dims[0];
    newShape.dims[dim++] = param.selfAttentionParam.headNum;
    newShape.dims[dim++] = param.headDim;
    newShape.dimNum = dim;
  };
  reshapeAndCacheNode.inTensorReshapeFuncs.at(0) = reshapeKVFunc;
  reshapeAndCacheNode.inTensorReshapeFuncs.at(1) = reshapeKVFunc;
  reshapeAndCacheNode.outTensorIds = {
      GetTensorIdx(tensorMap, "in_k_cache"),
      GetTensorIdx(tensorMap, "in_v_cache"),
  };
  opGraph.nodes.push_back(reshapeAndCacheNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status ConstructFusedInferAttentionNode(
    atb::Node &crossAttentionNode,
    const FusionAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap) {
  crossAttentionNode.inTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_q"),
      GetTensorIdx(tensorMap, "intermediate_k"),
      GetTensorIdx(tensorMap, "intermediate_v"),
      GetTensorIdx(tensorMap, "cross_kv_len"),
  };
  crossAttentionNode.operation =
      new atb_speed::common::FusedInferAttentionV2Operation(
          "FusedInferAttentionNode", param.aclnnFusedInferAttnParam);
  crossAttentionNode.inTensorReshapeFuncs.resize(
      crossAttentionNode.inTensorIds.size());
  auto reshapeKVFunc = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
    size_t dim = 0;
    newShape.dims[dim++] = oldShape.dims[0];
    newShape.dims[dim++] = oldShape.dims[1];
    newShape.dims[dim++] = param.selfAttentionParam.headNum;
    newShape.dims[dim++] = param.headDim;
    newShape.dimNum = dim;
  };
  crossAttentionNode.inTensorReshapeFuncs.at(1) = reshapeKVFunc;
  crossAttentionNode.inTensorReshapeFuncs.at(2) = reshapeKVFunc;
  auto reshapeQFunc = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
    size_t dim = 0;
    int batchSize = *param.bs;
    if (batchSize <= 0) {
      batchSize = 1;
    }
    newShape.dims[dim++] = batchSize;
    newShape.dims[dim++] = oldShape.dims[0] / batchSize;
    newShape.dims[dim++] = param.selfAttentionParam.headNum;
    newShape.dims[dim++] = param.headDim;
    newShape.dimNum = dim;
  };
  crossAttentionNode.inTensorReshapeFuncs.at(0) = reshapeQFunc;
  crossAttentionNode.outTensorIds = {
      GetTensorIdx(tensorMap, "intermediate_self_attention")};
  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddQScaleNode(const FusionAttentionParam<NormParamType> &param,
                          atb::GraphParam &opGraph,
                          std::map<std::string, uint32_t> &tensorMap) {
  atb::Node qScaleNode;
  atb::infer::ElewiseParam qScaleParam;
  qScaleParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
  qScaleParam.mulsParam.varAttr = 1.0 / sqrt(param.headDim);
  CREATE_OPERATION(qScaleParam, &qScaleNode.operation);
  qScaleNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
  qScaleNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
  opGraph.nodes.push_back(qScaleNode);
  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status Attention(const FusionAttentionParam<NormParamType> &param,
                      atb::Operation **operation) {
  atb::GraphParam opGraph;
  opGraph.name = "Attention";
  std::map<std::string, uint32_t> tensorMap;
  if (param.isOneRecEncoder || param.isOneRecDecoder) {
    tensorMap = ConstructOneRecTensorMap(param, opGraph.inTensorNum,
                                         opGraph.outTensorNum,
                                         opGraph.internalTensorNum,
                                         /*isEncoder=*/param.isOneRecEncoder);
  } else {
    tensorMap =
        ConstructTensorMap(param, opGraph.inTensorNum, opGraph.outTensorNum,
                           opGraph.internalTensorNum);
  }
  ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
  ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
  ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum "
                      << opGraph.internalTensorNum);

  if (param.layerLinearDescs.size() != 0 &&
      CheckParamVectorSize(param.layerLinearDescs, DENSE_LINEAR_INDEX + 1) !=
          atb::NO_ERROR) {
    ATB_SPEED_LOG_ERROR(
        "The size of param.layerLinearDescs is wrong, please check");
    return atb::ERROR_INVALID_PARAM;
  }
  if (param.layerLinearQuantType.size() != 0 &&
      CheckParamVectorSize(param.layerLinearQuantType,
                           DENSE_LINEAR_INDEX + 1) != atb::NO_ERROR) {
    ATB_SPEED_LOG_ERROR(
        "The size of param.layerLinearQuantType is wrong, please check");
    return atb::ERROR_INVALID_PARAM;
  }
  if (CheckParamVectorSize(param.layerLinearTransposeType,
                           DENSE_LINEAR_INDEX + 1) != atb::NO_ERROR) {
    ATB_SPEED_LOG_ERROR(
        "The size of param.layerLinearTransposeType is wrong, please check");
    return atb::ERROR_INVALID_PARAM;
  }

  if (param.enableRopeQuantKvcache) {
    // AddQNormLinearNode only, skip others
    CHECK_OPERATION_STATUS_RETURN(
        AddFAttnQKVLinearSplitNode(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(
        AddRopeQuantKvcacheOperation(opGraph, param, tensorMap));
  } else {
    // QKV Node
    CHECK_OPERATION_STATUS_RETURN(
        AddFAttnQKVLinearSplitNode(param, opGraph, tensorMap));

    // Rope Node
    if (param.rotaryType != RotaryType::NO_ROTARY && !param.enableSplitRmsNormRope) {
      CHECK_OPERATION_STATUS_RETURN(
          AddFAttnRopeNode(param, opGraph, tensorMap));
    }

    // QScale Node
    if (param.enableQScale) {
      CHECK_OPERATION_STATUS_RETURN(AddQScaleNode(param, opGraph, tensorMap));
    }

    bool atbAttentionDequant =
        param.attnBackend == atb_speed::common::OpBackend::ATB &&
        param.pageAttentionParam.quantType ==
            atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION;
    bool aclnnAttentionDequant =
        param.attnBackend == atb_speed::common::OpBackend::ACLNN &&
        param.aclnnIncreAttentionParam.hasKVQuant;
    if (atbAttentionDequant || aclnnAttentionDequant) {
      // K Quant
      CHECK_OPERATION_STATUS_RETURN(
          AddKVValueQuantNode(param, opGraph, tensorMap, true));
      // V Quant
      CHECK_OPERATION_STATUS_RETURN(
          AddKVValueQuantNode(param, opGraph, tensorMap, false));
    }
  }

  // FA3 QKV Quant Node
  if (!param.isPrefill &&
      param.pageAttentionParam.quantType ==
          atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
    CHECK_OPERATION_STATUS_RETURN(
        AddQKVQuantNode(param, opGraph, tensorMap, "Q"));
  }
  if (param.pageAttentionParam.quantType ==
      atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
    ATB_SPEED_LOG_DEBUG("Enter AddQKVQuantNode K");
    CHECK_OPERATION_STATUS_RETURN(
        AddQKVQuantNode(param, opGraph, tensorMap, "K"));
    ATB_SPEED_LOG_DEBUG("Enter AddQKVQuantNode V");
    CHECK_OPERATION_STATUS_RETURN(
        AddQKVQuantNode(param, opGraph, tensorMap, "V"));
  }

  const bool useOneRecDecoderFAS =
      param.isOneRecDecoder &&
      param.attnBackend == atb_speed::common::OpBackend::ACLNN &&
      param.aclnnFAScoreParam.headNum > 0 &&
      !(param.isPrefill && param.enableXattention);

  // OneRec decoder FAS: add Cast node to extract seqlen from mask shape.
  if (useOneRecDecoderFAS) {
    atb::Node castNode;
    atb_speed::common::AclNNCastParam castParam;
    castParam.dtype = ACL_FLOAT16;
    castNode.operation =
        new atb_speed::common::CastOperation("CastOperation", castParam);
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "in_attention_mask")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_mask")};
    castNode.inTensorReshapeFuncs.resize(castNode.inTensorIds.size());
    castNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape,
                                           atb::Dims &newShape) {
      newShape.dimNum = oldShape.dimNum;
      *param.seqlen = oldShape.dims[oldShape.dimNum - 1];
      if (oldShape.dimNum >= 3) {
        *param.bs = oldShape.dims[0];
      }
      for (int i = 0; i < newShape.dimNum; ++i) {
        newShape.dims[i] = oldShape.dims[i];
      }
    };
    opGraph.nodes.push_back(castNode);
  }

  // SelfAttention Node
  if (param.isFIA) {
    if (param.aclnnFusedInferAttnParam.inputLayout == "BSND") {
      CHECK_OPERATION_STATUS_RETURN(AddPadNode(opGraph, param, tensorMap));
    }
    CHECK_OPERATION_STATUS_RETURN(AddFIA(opGraph, param, tensorMap));
    if (param.aclnnFusedInferAttnParam.inputLayout == "BSND") {
      CHECK_OPERATION_STATUS_RETURN(AddUnpadNode(opGraph, param, tensorMap));
    }
  } else if (useOneRecDecoderFAS) {
    atb::Node selfAttentionNode;
    CHECK_OPERATION_STATUS_RETURN(
        ConstructFAScoreNode(selfAttentionNode, param, tensorMap));
    std::vector<std::string> attnOutTensorName = {
        "intermediate_self_attention", "softmax_max_out", "softmax_sum_out",
        "softmax_out_out"};
    selfAttentionNode.outTensorIds =
        GetTensorIdxList(tensorMap, attnOutTensorName);
    opGraph.nodes.push_back(selfAttentionNode);
  } else {
    CHECK_OPERATION_STATUS_RETURN(AddSelfAttention(opGraph, param, tensorMap));
  }

  // Dense Node
  CHECK_OPERATION_STATUS_RETURN(
      AddSelfOutLinearParallelNode(param, opGraph, tensorMap));

  opGraph.inferShapeFunc =
      [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
          atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(1).shape.dims[0];
        if (param.enableAddNorm) {
          outTensorDescs.at(1) = inTensorDescs.at(0);
        }
        return atb::NO_ERROR;
      };

  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
  return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status CrossAttention(const FusionAttentionParam<NormParamType> &param,
                           atb::Operation **operation) {
  atb::GraphParam opGraph;
  opGraph.name = "CrossAttention";
  std::map<std::string, uint32_t> tensorMap =
      ConstructOneRecCrossAttentionTensorMap(param, opGraph.inTensorNum,
                                             opGraph.outTensorNum,
                                             opGraph.internalTensorNum);
  if (param.layerLinearDescs.size() != 0 &&
      CheckParamVectorSize(param.layerLinearDescs, DENSE_LINEAR_INDEX + 1) !=
          atb::NO_ERROR) {
    ATB_SPEED_LOG_ERROR(
        "The size of param.layerLinearDescs is wrong, please check");
    return atb::ERROR_INVALID_PARAM;
  }
  if (param.layerLinearQuantType.size() != 0 &&
      CheckParamVectorSize(param.layerLinearQuantType,
                           DENSE_LINEAR_INDEX + 1) != atb::NO_ERROR) {
    ATB_SPEED_LOG_ERROR(
        "The size of param.layerLinearQuantType is wrong, please check");
    return atb::ERROR_INVALID_PARAM;
  }
  if (CheckParamVectorSize(param.layerLinearTransposeType,
                           DENSE_LINEAR_INDEX + 1) != atb::NO_ERROR) {
    ATB_SPEED_LOG_ERROR(
        "The size of param.layerLinearTransposeType is wrong, please check");
    return atb::ERROR_INVALID_PARAM;
  }

  CHECK_OPERATION_STATUS_RETURN(
      AddCrossAttnQKVProjectionNodes(param, opGraph, tensorMap));
  if (param.enableQScale) {
    CHECK_OPERATION_STATUS_RETURN(AddQScaleNode(param, opGraph, tensorMap));
  }
  if (param.isPrefill) {
    bool atbAttentionDequant =
        param.attnBackend == atb_speed::common::OpBackend::ATB &&
        param.pageAttentionParam.quantType ==
            atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION;
    bool aclnnAttentionDequant =
        param.attnBackend == atb_speed::common::OpBackend::ACLNN &&
        param.aclnnIncreAttentionParam.hasKVQuant;
    if (atbAttentionDequant || aclnnAttentionDequant) {
      CHECK_OPERATION_STATUS_RETURN(
          AddKVValueQuantNode(param, opGraph, tensorMap, true));
      CHECK_OPERATION_STATUS_RETURN(
          AddKVValueQuantNode(param, opGraph, tensorMap, false));
    }
    if (param.pageAttentionParam.quantType ==
        atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
      CHECK_OPERATION_STATUS_RETURN(
          AddQKVQuantNode(param, opGraph, tensorMap, "Q"));
      CHECK_OPERATION_STATUS_RETURN(
          AddQKVQuantNode(param, opGraph, tensorMap, "K"));
      CHECK_OPERATION_STATUS_RETURN(
          AddQKVQuantNode(param, opGraph, tensorMap, "V"));
    }
  } else if (param.pageAttentionParam.quantType ==
             atb::infer::PagedAttentionParam::QuantType::
                 TYPE_QUANT_QKV_ONLINE) {
    CHECK_OPERATION_STATUS_RETURN(
        AddQKVQuantNode(param, opGraph, tensorMap, "Q"));
  }

  if (DebugStopOneRecCrossLayer0AfterQkv(param)) {
    atb::Node debugOutNode;
    atb::infer::ElewiseParam debugOutParam;
    debugOutParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    debugOutParam.mulsParam.varAttr = 1.0f;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugOutNode.operation));
    debugOutNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
    debugOutNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(debugOutNode);

    const uint32_t encoderOutputIdx =
        atb_speed::common::GetTensorIdx(tensorMap, "in_encoder_output");
    opGraph.inferShapeFunc =
        [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
          outTensorDescs.at(0) = inTensorDescs.at(0);
          if (param.enableAddNorm) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
          }
          if (param.isPrefill && param.isOneRecCrossAttention) {
            uint32_t outIndex = param.enableAddNorm ? 2 : 1;
            outTensorDescs.at(outIndex) = inTensorDescs.at(encoderOutputIdx);
            outTensorDescs.at(outIndex + 1) = inTensorDescs.at(encoderOutputIdx);
          }
          return atb::NO_ERROR;
        };
    ATB_SPEED_LOG_ERROR(
        "OneRec cross debug: stop layer0 after cross QKV projection stage.");
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
  }

  if (DebugStopOneRecCrossLayer0AfterKvOnly(param)) {
    atb::Node debugOutNode;
    atb::infer::ElewiseParam debugOutParam;
    debugOutParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    debugOutParam.mulsParam.varAttr = 1.0f;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugOutNode.operation));
    debugOutNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_self_residual_out")};
    debugOutNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(debugOutNode);

    const uint32_t encoderOutputIdx =
        atb_speed::common::GetTensorIdx(tensorMap, "in_encoder_output");
    opGraph.inferShapeFunc =
        [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
          outTensorDescs.at(0) = inTensorDescs.at(0);
          if (param.enableAddNorm) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
          }
          if (param.isPrefill && param.isOneRecCrossAttention) {
            uint32_t outIndex = param.enableAddNorm ? 2 : 1;
            outTensorDescs.at(outIndex) = inTensorDescs.at(encoderOutputIdx);
            outTensorDescs.at(outIndex + 1) = inTensorDescs.at(encoderOutputIdx);
          }
          return atb::NO_ERROR;
        };
    ATB_SPEED_LOG_ERROR(
        "OneRec cross debug: stop layer0 after cross K/V projection stage.");
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
  }

  if (DebugStopOneRecCrossLayer0AfterKOnly(param)) {
    atb::Node debugOutNode;
    atb::infer::ElewiseParam debugOutParam;
    debugOutParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    debugOutParam.mulsParam.varAttr = 1.0f;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugOutNode.operation));
    debugOutNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_self_residual_out")};
    debugOutNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(debugOutNode);

    atb::Node debugVCacheNode;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugVCacheNode.operation));
    debugVCacheNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_encoder_output")};
    debugVCacheNode.outTensorIds = {GetTensorIdx(tensorMap, "in_cross_v_cache")};
    opGraph.nodes.push_back(debugVCacheNode);

    const uint32_t encoderOutputIdx =
        atb_speed::common::GetTensorIdx(tensorMap, "in_encoder_output");
    opGraph.inferShapeFunc =
        [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
          outTensorDescs.at(0) = inTensorDescs.at(0);
          if (param.enableAddNorm) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
          }
          if (param.isPrefill && param.isOneRecCrossAttention) {
            uint32_t outIndex = param.enableAddNorm ? 2 : 1;
            outTensorDescs.at(outIndex) = inTensorDescs.at(encoderOutputIdx);
            outTensorDescs.at(outIndex + 1) = inTensorDescs.at(encoderOutputIdx);
          }
          return atb::NO_ERROR;
        };
    ATB_SPEED_LOG_ERROR(
        "OneRec cross debug: stop layer0 after cross K projection stage.");
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
  }

  if (DebugStopOneRecCrossLayer0WithIdentityK(param)) {
    atb::Node debugOutNode;
    atb::infer::ElewiseParam debugOutParam;
    debugOutParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    debugOutParam.mulsParam.varAttr = 1.0f;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugOutNode.operation));
    debugOutNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_self_residual_out")};
    debugOutNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(debugOutNode);

    atb::Node debugKCacheNode;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugKCacheNode.operation));
    debugKCacheNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_encoder_output")};
    debugKCacheNode.outTensorIds = {GetTensorIdx(tensorMap, "in_cross_k_cache")};
    opGraph.nodes.push_back(debugKCacheNode);

    const uint32_t encoderOutputIdx =
        atb_speed::common::GetTensorIdx(tensorMap, "in_encoder_output");
    opGraph.inferShapeFunc =
        [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
          outTensorDescs.at(0) = inTensorDescs.at(0);
          if (param.enableAddNorm) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
          }
          if (param.isPrefill && param.isOneRecCrossAttention) {
            uint32_t outIndex = param.enableAddNorm ? 2 : 1;
            outTensorDescs.at(outIndex) = inTensorDescs.at(encoderOutputIdx);
            outTensorDescs.at(outIndex + 1) = inTensorDescs.at(encoderOutputIdx);
          }
          return atb::NO_ERROR;
        };
    ATB_SPEED_LOG_ERROR(
        "OneRec cross debug: stop layer0 with identity K/V cache outputs.");
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
  }

  if (DebugStopOneRecCrossLayer0WithIdentityV(param)) {
    atb::Node debugOutNode;
    atb::infer::ElewiseParam debugOutParam;
    debugOutParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    debugOutParam.mulsParam.varAttr = 1.0f;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugOutNode.operation));
    debugOutNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_self_residual_out")};
    debugOutNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(debugOutNode);

    atb::Node debugVCacheNode;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugVCacheNode.operation));
    debugVCacheNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_encoder_output")};
    debugVCacheNode.outTensorIds = {GetTensorIdx(tensorMap, "in_cross_v_cache")};
    opGraph.nodes.push_back(debugVCacheNode);

    const uint32_t encoderOutputIdx =
        atb_speed::common::GetTensorIdx(tensorMap, "in_encoder_output");
    opGraph.inferShapeFunc =
        [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
          outTensorDescs.at(0) = inTensorDescs.at(0);
          if (param.enableAddNorm) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
          }
          if (param.isPrefill && param.isOneRecCrossAttention) {
            uint32_t outIndex = param.enableAddNorm ? 2 : 1;
            outTensorDescs.at(outIndex) = inTensorDescs.at(encoderOutputIdx);
            outTensorDescs.at(outIndex + 1) = inTensorDescs.at(encoderOutputIdx);
          }
          return atb::NO_ERROR;
        };
    ATB_SPEED_LOG_ERROR(
        "OneRec cross debug: stop layer0 with identity V cache output.");
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
  }

  if (DebugStopOneRecCrossLayer0WithIdentityKv(param)) {
    atb::Node debugOutNode;
    atb::infer::ElewiseParam debugOutParam;
    debugOutParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    debugOutParam.mulsParam.varAttr = 1.0f;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugOutNode.operation));
    debugOutNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_self_residual_out")};
    debugOutNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(debugOutNode);

    atb::Node debugKCacheNode;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugKCacheNode.operation));
    debugKCacheNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_encoder_output")};
    debugKCacheNode.outTensorIds = {GetTensorIdx(tensorMap, "in_cross_k_cache")};
    opGraph.nodes.push_back(debugKCacheNode);

    atb::Node debugVCacheNode;
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(debugOutParam, &debugVCacheNode.operation));
    debugVCacheNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_encoder_output")};
    debugVCacheNode.outTensorIds = {GetTensorIdx(tensorMap, "in_cross_v_cache")};
    opGraph.nodes.push_back(debugVCacheNode);

    const uint32_t encoderOutputIdx =
        atb_speed::common::GetTensorIdx(tensorMap, "in_encoder_output");
    opGraph.inferShapeFunc =
        [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
          outTensorDescs.at(0) = inTensorDescs.at(0);
          if (param.enableAddNorm) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
          }
          if (param.isPrefill && param.isOneRecCrossAttention) {
            uint32_t outIndex = param.enableAddNorm ? 2 : 1;
            outTensorDescs.at(outIndex) = inTensorDescs.at(encoderOutputIdx);
            outTensorDescs.at(outIndex + 1) = inTensorDescs.at(encoderOutputIdx);
          }
          return atb::NO_ERROR;
        };
    ATB_SPEED_LOG_ERROR(
        "OneRec cross debug: stop layer0 with identity K/V outputs.");
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
  }

  CHECK_OPERATION_STATUS_RETURN(AddSelfAttention(opGraph, param, tensorMap));
  CHECK_OPERATION_STATUS_RETURN(
      AddCrossOutLinearParallelNode(param, opGraph, tensorMap));

  const uint32_t encoderOutputIdx =
      atb_speed::common::GetTensorIdx(tensorMap, "in_encoder_output");
  opGraph.inferShapeFunc =
      [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
          atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.enableAddNorm) {
          outTensorDescs.at(1) = inTensorDescs.at(0);
        }
        if (param.isPrefill && param.isOneRecCrossAttention) {
          uint32_t outIndex = param.enableAddNorm ? 2 : 1;
          outTensorDescs.at(outIndex) = inTensorDescs.at(encoderOutputIdx);
          outTensorDescs.at(outIndex + 1) = inTensorDescs.at(encoderOutputIdx);
        }
        return atb::NO_ERROR;
      };
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
  return atb::NO_ERROR;
}

template atb::Status ConstructAttentionQuantTensorMap(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, std::vector<std::string>> &attnInTensorCandidates,
    std::map<std::string, std::vector<std::string>>
        &attnIntermediateTensorCandidates,
    std::vector<std::string> &inTensorList,
    std::vector<std::string> &intermediateTensorList);
template std::map<std::string, uint32_t>
ConstructTensorMap(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
                   uint32_t &inTensorNum, uint32_t &outTensorNum,
                   uint32_t &internalTensorNum);
template atb::Status
AddFAttnRopeNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
                 atb::GraphParam &opGraph,
                 std::map<std::string, uint32_t> &tensorMap);
template atb::Status
AddKVValueQuantNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
                    atb::GraphParam &opGraph,
                    std::map<std::string, uint32_t> &tensorMap, bool isK);
template atb::Status AddSelfOutLinearParallelNode(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status
AddQScaleNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
              atb::GraphParam &opGraph,
              std::map<std::string, uint32_t> &tensorMap);
template atb::Status
Attention(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
          atb::Operation **operation);
template atb::Status
CrossAttention(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
               atb::Operation **operation);

template atb::Status ConstructAttentionQuantTensorMap(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, std::vector<std::string>> &attnInTensorCandidates,
    std::map<std::string, std::vector<std::string>>
        &attnIntermediateTensorCandidates,
    std::vector<std::string> &inTensorList,
    std::vector<std::string> &intermediateTensorList);
template std::map<std::string, uint32_t> ConstructTensorMap(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template atb::Status
AddFAttnRopeNode(const FusionAttentionParam<atb::infer::LayerNormParam> &param,
                 atb::GraphParam &opGraph,
                 std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddKVValueQuantNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap,
    bool isK);
template atb::Status AddSelfOutLinearParallelNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status
AddQScaleNode(const FusionAttentionParam<atb::infer::LayerNormParam> &param,
              atb::GraphParam &opGraph,
              std::map<std::string, uint32_t> &tensorMap);
template atb::Status
Attention(const FusionAttentionParam<atb::infer::LayerNormParam> &param,
          atb::Operation **operation);
template atb::Status
CrossAttention(const FusionAttentionParam<atb::infer::LayerNormParam> &param,
               atb::Operation **operation);
} // namespace common
} // namespace atb_speed
