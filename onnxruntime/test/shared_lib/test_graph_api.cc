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
                    const gsl::span<OrtOpAttr**> attributes = {},
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
    OrtShape* input_shape = nullptr;
    std::vector<int64_t> input_dims = {3, 2};
    Ort::ThrowOnError(graph_api.CreateFixedShape(input_dims.data(), input_dims.size(), &input_shape));

    OrtValueInfo* input_info = nullptr;
    Ort::ThrowOnError(graph_api.CreateTensorValueInfo("X", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_shape,
                                                      &input_info));
    ASSERT_EQ(input_shape, nullptr) << "CreateTensorValueInfo should take ownership of input_shape";

    // model outputs
    OrtShape* output_shape = nullptr;
    std::vector<int64_t> output_dims = {3, 3};
    Ort::ThrowOnError(graph_api.CreateFixedShape(output_dims.data(), output_dims.size(), &output_shape));

    OrtValueInfo* output_info = nullptr;
    Ort::ThrowOnError(graph_api.CreateTensorValueInfo("Z", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &output_shape,
                                                      &output_info));
    ASSERT_EQ(output_shape, nullptr) << "CreateTensorValueInfo should take ownership of output_shape";

    Ort::ThrowOnError(graph_api.AddInput(graph, &input_info));
    ASSERT_EQ(input_info, nullptr) << "AddInput should take ownership of input_info";

    Ort::ThrowOnError(graph_api.AddOutput(graph, &output_info));
    ASSERT_EQ(output_info, nullptr) << "AddOutput should take ownership of output_info";

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
    std::vector<OrtOpAttr**> node_attributes = {&alpha_attr};
    OrtNode* node = CreateNode(graph_api, "Gemm", "Gemm1", node_input_names, node_output_names, node_attributes);

    ASSERT_EQ(alpha_attr, nullptr) << "CreateNode should take ownership of the attributes";

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
      weights.emplace_back(std::make_unique<std::vector<float>>(
          std::initializer_list<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
      auto& y_values = *weights.back();
      auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

      // if you use this API the initializer data MUST remain valid for the lifetime of the InferenceSession
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

  const auto build_model = [&](GraphApi::Model& model) -> void {
    Ort::GraphApi::Graph graph;

    //
    // Create OrtModel with a Gemm. X input is 3x2, Y input is 2x3, Z output is 3x3.
    // X is model input. Y is initializer.
    // Set the alpha attribute of the Gemm node to 2.0 to test attribute handling.
    //

    // model input
    std::vector<int64_t> input_dims({3, 2});
    GraphApi::Shape input_shape(input_dims);

    auto input_info = GraphApi::ValueInfo::CreateTensorValueInfo(std::string("X"), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                                 input_shape);

    // model outputs
    std::vector<int64_t> output_dims = {3, 3};
    GraphApi::Shape output_shape(output_dims);
    auto output_info = GraphApi::ValueInfo::CreateTensorValueInfo(std::string("Z"), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                                  output_shape);

    graph.AddInput(input_info);
    graph.AddOutput(output_info);

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
    model = GraphApi::Model(opsets);
    model.AddGraph(graph);

    ASSERT_EQ(input_shape, nullptr) << "ValueInfo should take ownership of input_shape";
    ASSERT_EQ(output_shape, nullptr) << "ValueInfo should take ownership of output_shape";
    ASSERT_EQ(input_info, nullptr) << "AddInput should take ownership of input_info";
    ASSERT_EQ(output_info, nullptr) << "AddOutput should take ownership of output_info";
    ASSERT_EQ(attributes[0], nullptr) << "Node should take ownership of the attributes";
    ASSERT_EQ(node, nullptr) << "AddNode should take ownership of the node";
    ASSERT_EQ(graph, nullptr) << "AddGraph should take ownership of the graph";
  };

  GraphApi::Model model(nullptr);
  build_model(model);

  ASSERT_NE(model, nullptr) << "build_model should have created a model";

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

// dynamic shape

// Constant node

// multiple nodes to test shape inferencing between nodes
