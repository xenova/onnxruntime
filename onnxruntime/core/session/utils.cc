// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/utils.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_c_api.h"

using namespace onnxruntime;

common::Status CopyStringToOutputArg(std::string_view str, const char* err_msg, char* out, size_t* size) {
  const size_t str_len = str.size();
  const size_t req_size = str_len + 1;

  if (out == nullptr) {  // User is querying the total output buffer size
    *size = req_size;
    return onnxruntime::common::Status::OK();
  }

  if (*size >= req_size) {  // User provided a buffer of sufficient size
    std::memcpy(out, str.data(), str_len);
    out[str_len] = '\0';
    *size = req_size;
    return onnxruntime::common::Status::OK();
  }

  // User has provided a buffer that is not large enough
  *size = req_size;
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, err_msg);
}

OrtStatus* InitializeSession(_In_ const OrtSessionOptions* options,
                             _In_ std::unique_ptr<::onnxruntime::InferenceSession>& sess,
                             _Inout_opt_ OrtPrepackedWeightsContainer* prepacked_weights_container) {
  // we need to disable mem pattern if DML is one of the providers since DML doesn't have the concept of
  // byte addressable memory
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list;
  if (options) {
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider();
      provider_list.push_back(std::move(provider));
    }
  }

  // register the providers
  for (auto& provider : provider_list) {
    if (provider) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->RegisterExecutionProvider(std::move(provider)));
    }
  }

  if (prepacked_weights_container != nullptr) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->AddPrePackedWeightsContainer(
        reinterpret_cast<PrepackedWeightsContainer*>(prepacked_weights_container)));
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Initialize());

  return nullptr;
}
