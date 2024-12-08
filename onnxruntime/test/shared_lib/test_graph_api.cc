// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #include <algorithm>
// #include <atomic>
// #include <fstream>
// #include <iostream>
// #include <memory>
// #include <mutex>
// #include <sstream>
// #include <thread>
// #include <vector>
//
// #include <absl/base/config.h>
#include <gsl/gsl>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

// #include "core/common/common.h"
// #include "core/common/narrow.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
// #include "core/session/onnxruntime_run_options_config_keys.h"
// #include "core/util/thread_utils.h"
//
// #include "test/shared_lib/custom_op_utils.h"
#include "test/shared_lib/test_fixture.h"
#include "test/shared_lib/utils.h"
// #include "test/util/include/providers.h"
// #include "test/util/include/test_allocator.h"

#include "onnxruntime_config.h"  // generated file in build output dir

namespace {
const OrtApi& GetOrtApi() {
  return *Ort::Global<void>::api_;
}

const OrtGraphApi& GetGraphApi() {
  return *Ort::Global<void>::api_->GetGraphApi();
}

template <typename ModelOutputT, typename ModelInputT = float>
void TestInference(Ort::Env& env, const OrtModel& graph_api_model,
                   const std::vector<ModelInputT>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<ModelOutputT>& expected_values_y,
                   Ort::SessionOptions* session_options_for_test = nullptr) {
  Ort::SessionOptions default_session_options;
  Ort::SessionOptions& session_options = session_options_for_test ? *session_options_for_test
                                                                  : default_session_options;

  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  // without preallocated output tensor
  RunSession<ModelOutputT, ModelInputT>(default_allocator.get(),
                                        session,
                                        inputs,
                                        output_name,
                                        expected_dims_y,
                                        expected_values_y,
                                        nullptr);
}

OrtNode* CreateNode(const OrtGraphApi& api,
                    const char* operator_name, const char* node_name,
                    const gsl::span<const char*> input_names,
                    const gsl::span<const char*> output_names,
                    const gsl::span<OrtOpAttr*> attributes = {},
                    const char* domain_name = onnxruntime::kOnnxDomain) {
  OrtNode* node = nullptr;
  Ort::ThrowOnError(api.CreateNode(operator_name, domain_name, node_name,
                                   input_names.data(), input_names.size(),
                                   output_names.data(), output_names.size(),
                                   attributes.data(), attributes.size(),
                                   &node));
  return node;
}

}  // namespace

// todo: run with initializer and Constant node
TEST(GraphApiTest, Basic) {
  // return void so we can use ASSERT_* in the lambda
  const auto build_model = [](bool use_constant_node, OrtModel*& model) -> void {
    const auto& api = GetOrtApi();
    const auto& graph_api = GetGraphApi();

    OrtGraph* graph = nullptr;
    Ort::ThrowOnError(graph_api.CreateGraph(&graph));

    //
    // Create OrtModel with a Gemm. X input is 3x2, Y input is 2x3, Z output is 3x3.
    // X is model input. Y is initializer.
    // Set the alpha attribute of the Gemm node to 2.0 to test attribute handling.
    //

    // model input
    OrtShape* input_shape = nullptr;
    std::vector<int64_t> input_dims = {3, 2};
    Ort::ThrowOnError(graph_api.CreateFixedShape(input_dims.data(), input_dims.size(), &input_shape));

    OrtValueInfo* input_info = nullptr;
    Ort::ThrowOnError(graph_api.CreateTensorValueInfo("X", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_shape, &input_info));
    ASSERT_EQ(input_shape, nullptr) << "CreateTensorValueInfo should take ownership of input_shape";

    // model outputs
    OrtShape* output_shape = nullptr;
    std::vector<int64_t> output_dims = {3, 3};
    Ort::ThrowOnError(graph_api.CreateFixedShape(output_dims.data(), output_dims.size(), &output_shape));

    //
    // Gemm node
    //

    // add attribute to test it works
    // TODO: It's slightly ugly to have to use the ORT API to create the attribute using the existing CreateOpAttr,
    // but we can hide that in the C++ wrapper classes.
    OrtOpAttr* alpha_attr = nullptr;
    float alpha_value = 2.0;
    Ort::ThrowOnError(api.CreateOpAttr("alpha", &alpha_value, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT, &alpha_attr));

    // nodes
    std::vector<const char*> node_input_names = {"X", "Y"};
    std::vector<const char*> node_output_names = {"Z"};
    std::vector<OrtOpAttr*> node_attributes = {alpha_attr};
    OrtNode* node = CreateNode(graph_api, "Gemm", "Gemm1", node_input_names, node_output_names, node_attributes);

    for (auto* attr : node_attributes) {
      ASSERT_EQ(attr, nullptr) << "CreateNode should take ownership of the attributes";
    }

    Ort::ThrowOnError(graph_api.AddNode(graph, &node));
    ASSERT_EQ(node, nullptr) << "AddNode should take ownership of the node";

    if (use_constant_node) {
      // create an attribute for the Y input
      // create Constant node that produces "Y" output with the value_floats attribute
      ASSERT_FALSE(true) << "Not implemented";
    } else {
      // create an initializer for the Y input
      OrtValue* y_tensor = nullptr;
      std::vector<int64_t> y_dims = {2, 3};
      std::vector<float> y_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

      Ort::ThrowOnError(api.CreateTensorWithDataAsOrtValue(info,
                                                           y_values.data(), y_values.size() * sizeof(y_values[0]),
                                                           y_dims.data(), y_dims.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &y_tensor));
      Ort::ThrowOnError(graph_api.AddInitializer(graph, "Y", &y_tensor));

      std::vector<const char*> domain_names = {onnxruntime::kOnnxDomain};
      std::vector<int> opset_versions = {18};
      Ort::ThrowOnError(graph_api.CreateModel(domain_names.data(), opset_versions.data(), domain_names.size(),
                                              &model));
      Ort::ThrowOnError(graph_api.AddGraph(model, &graph));
      ASSERT_EQ(graph, nullptr) << "AddGraph should take ownership of the graph";
    }
  };

  OrtModel* model = nullptr;
  build_model(false, model);

  ASSERT_NE(model, nullptr) << "build_model should have created a model";

  // simple inference test
  // prepare inputs
  // std::vector<Input>
  //    inputs(1);
  // Input& input = inputs.back();
  // input.name = "X";
  // input.dims = {3, 2};
  // input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  //// prepare expected inputs and outputs
  // std::vector<int64_t> expected_dims_y = {3, 2};
  // std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  // TestInference<float>(*ort_env, MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, GetParam(),
  //                      nullptr, nullptr);
}

// dynamic shape

// Constant node

// multiple nodes to test shape inferencing between nodes
