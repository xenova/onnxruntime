// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>
#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"

onnxruntime::common::Status CopyStringToOutputArg(std::string_view str, const char* err_msg, char* out, size_t* size);

struct OrtSessionOptions;
struct OrtStatus;
struct OrtPrepackedWeightsContainer;
namespace onnxruntime {
class InferenceSession;
}

OrtStatus* InitializeSession(_In_ const OrtSessionOptions* options,
                             _In_ std::unique_ptr<::onnxruntime::InferenceSession>& sess,
                             _Inout_opt_ OrtPrepackedWeightsContainer* prepacked_weights_container = nullptr);
