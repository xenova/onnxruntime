// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gsl/gsl>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/graph/constants.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/shared_lib/test_fixture.h"
#include "test/shared_lib/utils.h"
#include "test/util/include/test_allocator.h"

#include "onnxruntime_config.h"  // generated file in build output dir

extern std::unique_ptr<Ort::Env> ort_env;

using namespace Ort;

namespace {
template <typename ModelOutputT, typename ModelInputT = float>
void TestInference(Ort::Env& env,
                   GraphApi::Model& graph_api_model,
                   const std::vector<Input>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims,
                   const std::vector<ModelOutputT>& expected_values,
                   Ort::SessionOptions* session_options_for_test = nullptr) {
  Ort::SessionOptions default_session_options;
  Ort::SessionOptions& session_options = session_options_for_test ? *session_options_for_test
                                                                  : default_session_options;

  // save model if you want to debug
  // session_options.SetOptimizedModelFilePath(ORT_TSTR("graph_api_model.onnx"));

  Ort::Session session(env, graph_api_model, session_options);

  // Session should not require the model to stay alive so free it now to test.
  graph_api_model = GraphApi::Model(nullptr);

  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  // without preallocated output tensor
  RunSession<ModelOutputT, ModelInputT>(default_allocator.get(),
                                        session,
                                        inputs,
                                        output_name,
                                        expected_dims,
                                        expected_values,
                                        nullptr);
}

// Create OrtNode using the C API
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

// Test the GraphApi C api
// Uses the ORT C++ api for the rest for simplicity
TEST(GraphApiTest, Basic_CApi) {
  const auto& api = Ort::GetApi();
  const auto& graph_api = Ort::GetGraphApi();

  // initializers that are used directly by the model. as there's no copy they must remain valid
  std::vector<std::unique_ptr<std::vector<float>>> weights;

  // return void so we can use ASSERT_* in the lambda
  const auto build_model = [&](bool use_constant_node, OrtModel*& model) -> void {
    OrtGraph* graph = nullptr;
    Ort::ThrowOnError(graph_api.CreateGraph(&graph));

    //
    // Create OrtModel with a Gemm. X input is 3x2, Y input is 2x3, Z output is 3x3.
    // X is model input. Y is initializer.
    // Set the alpha attribute of the Gemm node to 2.0 to test attribute handling.
    //

    // model input
    OrtTensorTypeAndShapeInfo* tensor_type_info = nullptr;
    std::vector<int64_t> input_dims = {3, 2};
    // can use api.SetSymbolicDimensions to set symbolic dimensions.
    // the input array should have the same rank as the call to SetDimensions.
    // e.g. call SetDimensions with {-1, 3, 2} and SetSymbolicDimensions with {"N", nullptr, nullptr} to create
    //      a shape of {"N", 3, 2}

    Ort::ThrowOnError(api.CreateTensorTypeAndShapeInfo(&tensor_type_info));
    Ort::ThrowOnError(api.SetTensorElementType(tensor_type_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    Ort::ThrowOnError(api.SetDimensions(tensor_type_info, input_dims.data(), input_dims.size()));

    OrtTypeInfo* input_type_info = nullptr;
    Ort::ThrowOnError(api.CreateTensorTypeInfo(tensor_type_info, &input_type_info));
    api.ReleaseTensorTypeAndShapeInfo(tensor_type_info);  // input_type_info took a copy

    // create ValueInfo and release the type info as CreateValueInfo takes a copy.
    OrtValueInfo* input_value_info = nullptr;
    Ort::ThrowOnError(graph_api.CreateValueInfo("X", input_type_info, &input_value_info));
    api.ReleaseTypeInfo(input_type_info);  // input_value_info took a copy
    tensor_type_info = nullptr;

    // model outputs
    OrtTypeInfo* output_type_info = nullptr;
    std::vector<int64_t> output_dims = {3, 3};

    Ort::ThrowOnError(api.CreateTensorTypeAndShapeInfo(&tensor_type_info));
    Ort::ThrowOnError(api.SetTensorElementType(tensor_type_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    Ort::ThrowOnError(api.SetDimensions(tensor_type_info, output_dims.data(), output_dims.size()));

    Ort::ThrowOnError(api.CreateTensorTypeInfo(tensor_type_info, &output_type_info));
    api.ReleaseTensorTypeAndShapeInfo(tensor_type_info);  // input_type_info took a copy

    OrtValueInfo* output_value_info = nullptr;
    Ort::ThrowOnError(graph_api.CreateValueInfo("Z", output_type_info, &output_value_info));
    api.ReleaseTypeInfo(output_type_info);

    std::vector<OrtValueInfo*> graph_inputs = {input_value_info};
    std::vector<OrtValueInfo*> graph_outputs = {output_value_info};
    Ort::ThrowOnError(graph_api.SetGraphInputs(graph, graph_inputs.data(), graph_inputs.size()));
    Ort::ThrowOnError(graph_api.SetGraphOutputs(graph, graph_outputs.data(), graph_outputs.size()));

    //
    // Gemm node
    //

    OrtOpAttr* alpha_attr = nullptr;
    float alpha_value = 2.0;
    Ort::ThrowOnError(api.CreateOpAttr("alpha", &alpha_value, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT, &alpha_attr));

    std::vector<const char*> node_input_names = {"X", "Y"};
    std::vector<const char*> node_output_names = {"Z"};
    std::vector<OrtOpAttr*> node_attributes{alpha_attr};
    OrtNode* node = CreateNode(graph_api, "Gemm", "Gemm1", node_input_names, node_output_names, node_attributes);

    api.ReleaseOpAttr(alpha_attr);  // CreateNode copies an OrtOpAttr instances

    Ort::ThrowOnError(graph_api.AddNodeToGraph(graph, node));
    node = nullptr;  // graph now owns node

    if (use_constant_node) {
      // create an attribute for the Y input
      // create Constant node that produces "Y" output with the value_floats attribute
      ASSERT_FALSE(true) << "Not implemented";
    } else {
      // create an initializer for the Y input. add to `weights` so the memory remains valid
      OrtValue* y_tensor = nullptr;
      std::vector<int64_t> y_dims = {2, 3};
      weights.emplace_back(std::make_unique<std::vector<float>>(std::initializer_list<float>{1.0f, 2.0f, 3.0f,
                                                                                             4.0f, 5.0f, 6.0f}));
      auto& y_values = *weights.back();
      auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

      // if you use this API the initializer data MUST remain valid for the lifetime of the InferenceSession
      Ort::ThrowOnError(api.CreateTensorWithDataAsOrtValue(info,
                                                           y_values.data(), y_values.size() * sizeof(y_values[0]),
                                                           y_dims.data(), y_dims.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &y_tensor));
      Ort::ThrowOnError(graph_api.AddInitializerToGraph(graph, "Y", y_tensor));
      y_tensor = nullptr;  // graph now owns

      std::vector<const char*> domain_names = {onnxruntime::kOnnxDomain};
      std::vector<int> opset_versions = {18};
      Ort::ThrowOnError(graph_api.CreateModel(domain_names.data(), opset_versions.data(), domain_names.size(),
                                              &model));
      Ort::ThrowOnError(graph_api.AddGraphToModel(model, graph));
      graph = nullptr;  // model now owns
    }
  };

  OrtModel* model = nullptr;
  build_model(false, model);

  ASSERT_NE(model, nullptr) << "build_model should have created a model";

  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  std::vector<int64_t> expected_dims = {3, 3};
  GraphApi::Model cxx_model(model);
  TestInference<float>(*ort_env, cxx_model, inputs, "Z", expected_dims,
                       {18.0f, 24.0f, 30.0f,
                        38.0f, 52.0f, 66.0f,
                        58.0f, 80.0f, 102.0f});
}

TEST(GraphApiTest, Basic_CxxApi) {
  // initializers that are used directly by the model. as there's no copy they must remain valid
  std::vector<std::unique_ptr<std::vector<float>>> weights;

  Ort::GraphApi::Graph graph;

  //
  // Create OrtModel with a Gemm. X input is 3x2, Y input is 2x3, Z output is 3x3.
  // X is model input. Y is initializer.
  // Set the alpha attribute of the Gemm node to 2.0 to test attribute handling.
  //

  std::vector<GraphApi::ValueInfo> graph_inputs;
  std::vector<GraphApi::ValueInfo> graph_outputs;

  // model input
  std::vector<int64_t> input_dims({3, 2});
  TensorTypeAndShapeInfo input_tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_dims);
  auto input_type_info = TypeInfo::CreateTensorInfo(input_tensor_info.GetConst());
  graph_inputs.emplace_back("X", input_type_info.GetConst());

  // model outputs
  std::vector<int64_t> output_dims = {3, 3};
  TensorTypeAndShapeInfo output_tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_dims);
  auto output_type_info = TypeInfo::CreateTensorInfo(output_tensor_info.GetConst());
  graph_outputs.emplace_back("Z", output_type_info.GetConst());

  graph.SetInputs(graph_inputs);
  graph.SetOutputs(graph_outputs);

  //
  // Gemm node
  //

  std::vector<OpAttr> attributes;
  float alpha_value = 2.0;
  attributes.push_back(OpAttr("alpha", &alpha_value, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT));

  GraphApi::Node node("Gemm", onnxruntime::kOnnxDomain, "Gemm1", {"X", "Y"}, {"Z"}, attributes);

  graph.AddNode(node);

  // create an initializer for the Y input.
  // add to `weights` so it remains valid for the lifetime of the session and we can avoid copying the data.
  std::vector<int64_t> y_dims = {2, 3};
  weights.emplace_back(std::make_unique<std::vector<float>>(std::initializer_list<float>{1.0f, 2.0f, 3.0f,
                                                                                         4.0f, 5.0f, 6.0f}));
  auto& y_values = *weights.back();
  auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // if you use this API the initializer data MUST remain valid for the lifetime of the InferenceSession
  auto y_tensor = Value::CreateTensor(info, y_values.data(), y_values.size(), y_dims.data(), y_dims.size());
  graph.AddInitializer("Y", y_tensor);

  std::vector<GraphApi::Model::DomainOpsetPair> opsets{{onnxruntime::kOnnxDomain, 18}};
  GraphApi::Model model(opsets);
  model.AddGraph(graph);

  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  std::vector<int64_t> expected_dims = {3, 3};
  TestInference<float>(*ort_env, model, inputs, "Z", expected_dims,
                       {18.0f, 24.0f, 30.0f,
                        38.0f, 52.0f, 66.0f,
                        58.0f, 80.0f, 102.0f});
}
