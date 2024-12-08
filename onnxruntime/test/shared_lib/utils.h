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

template <typename ModelOutputT, typename ModelInputT = float, typename InputDataT = ModelInputT>
void RunSession(OrtAllocator* allocator, Ort::Session& session_object,
                const std::vector<InputDataT>& inputs,
                const char* output_name,
                const std::vector<int64_t>& dims_y,
                const std::vector<ModelOutputT>& values_y,
                Ort::Value* output_tensor);

