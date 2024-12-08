// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"
#include <limits>

OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream) {
  OrtCUDAProviderOptions cuda_options;

  cuda_options.device_id = 0;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
  cuda_options.gpu_mem_limit = std::numeric_limits<size_t>::max();
  cuda_options.arena_extend_strategy = 0;
  cuda_options.do_copy_in_default_stream = true;
  cuda_options.has_user_compute_stream = cuda_compute_stream != nullptr ? 1 : 0;
  cuda_options.user_compute_stream = cuda_compute_stream;
  cuda_options.default_memory_arena_cfg = nullptr;

  return cuda_options;
}

template <typename ModelOutputT, typename ModelInputT, typename InputDataT>
void RunSession(OrtAllocator* allocator, Ort::Session& session_object,
                const std::vector<InputDataT>& inputs,
                const char* output_name,
                const std::vector<int64_t>& dims_y,
                const std::vector<ModelOutputT>& values_y,
                Ort::Value* output_tensor) {
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(
        Ort::Value::CreateTensor(allocator->Info(allocator), const_cast<InputDataT*>(inputs[i].values.data()),
                                 inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }

  std::vector<Ort::Value> ort_outputs;
  if (output_tensor)
    session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                       &output_name, output_tensor, 1);
  else {
    ort_outputs = session_object.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                     &output_name, 1);
    ASSERT_EQ(ort_outputs.size(), 1u);
    output_tensor = &ort_outputs[0];
  }

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), dims_y);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), total_len);

  OutT* f = output_tensor->GetTensorMutableData<OutT>();
  for (size_t i = 0; i != total_len; ++i) {
    if constexpr (std::is_same<ModelOutputT, float>::value || std::is_same<ModelOutputT, double>::value) {
      ASSERT_NEAR(values_y[i], f[i], 1e-3);
    } else {
      ASSERT_EQ(values_y[i], f[i]);
    }
  }
}
