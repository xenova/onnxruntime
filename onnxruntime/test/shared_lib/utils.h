// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_cxx_api.h"

OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream = nullptr);

struct Input {
  const char* name = nullptr;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

template <typename ModelOutputT, typename ModelInputT = float, typename InputT = Input>
void RunSession(OrtAllocator* allocator, Ort::Session& session_object,
                const std::vector<InputT>& inputs,
                const char* output_name,
                const std::vector<int64_t>& dims_y,
                const std::vector<ModelOutputT>& values_y,
                Ort::Value* output_tensor) {
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(
        Ort::Value::CreateTensor(allocator->Info(allocator), const_cast<ModelInputT*>(inputs[i].values.data()),
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

  auto* f = output_tensor->GetTensorMutableData<ModelOutputT>();
  for (size_t i = 0; i != total_len; ++i) {
    if constexpr (std::is_same<ModelOutputT, float>::value || std::is_same<ModelOutputT, double>::value) {
      ASSERT_NEAR(values_y[i], f[i], 1e-3);
    } else {
      ASSERT_EQ(values_y[i], f[i]);
    }
  }
}
