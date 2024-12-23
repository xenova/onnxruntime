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

Ort::Session CreateSession(Ort::Env& env,
                           ModelBuilderAPI::Model& graph_api_model,
                           Ort::SessionOptions* session_options_for_test = nullptr) {
  Ort::SessionOptions default_session_options;
  Ort::SessionOptions& session_options = session_options_for_test ? *session_options_for_test
                                                                  : default_session_options;

  // Set this to save the model if you want to debug.
  // session_options.SetOptimizedModelFilePath(ORT_TSTR("model_builder_output.onnx"));

  Ort::Session session(env, graph_api_model, session_options);

  // Session should not require the model to stay alive so free it now to validate.
  graph_api_model = ModelBuilderAPI::Model(nullptr);

  return session;
}

template <typename ModelOutputT, typename ModelInputT = float>
void TestInference(Ort::Session& session,
                   const std::vector<Input<ModelInputT>>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims,
                   const std::vector<ModelOutputT>& expected_values) {
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
OrtNode* CreateNode(const OrtModelBuilderApi& api,
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

struct TestAllocator : public OrtAllocator {
  TestAllocator() {
    version = ORT_API_VERSION;
    Info = [](const struct OrtAllocator* this_ptr) -> const struct OrtMemoryInfo* {
      auto* test_allocator = static_cast<const TestAllocator*>(this_ptr);
      return test_allocator->memory_info;
    };

    Free = [](struct OrtAllocator* allocator, void* p) -> void {
      auto* test_allocator = static_cast<TestAllocator*>(allocator);
      // find the matching pointer and remove it
      auto it = std::find_if(test_allocator->weights.begin(), test_allocator->weights.end(),
                             [p](const std::unique_ptr<std::vector<float>>& v) { return v->data() == p; });
      if (it == test_allocator->weights.end()) {
        throw std::exception("Free called with unknown pointer");
      }

      test_allocator->weights.erase(it);
    };

    Alloc = [](struct OrtAllocator* /*this*/, size_t /*size*/) -> void* {
      throw std::exception("This should not be used");
    };

    Reserve = [](struct OrtAllocator* /*this*/, size_t /*size*/) -> void* {
      throw std::exception("This should not be used");
    };
  }

  // initializers that are used directly by the model. as there's no copy they must remain valid.
  // we store them in the test allocator so we can validate that Free is called
  std::vector<std::unique_ptr<std::vector<float>>> weights;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                                           OrtMemType::OrtMemTypeDefault);
};

// Test the ModelBuilderAPI C api
// Uses the ORT C++ api for the rest for simplicity
TEST(ModelBuilderAPITest, Basic_CApi) {
  const auto& api = Ort::GetApi();
  const auto& graph_api = Ort::GetModelBuilderApi();

  TestAllocator deleter;

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

    api.ReleaseOpAttr(alpha_attr);  // CreateNode copies all OrtOpAttr instances

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
      deleter.weights.emplace_back(
          std::make_unique<std::vector<float>>(std::initializer_list<float>{1.0f, 2.0f, 3.0f,
                                                                            4.0f, 5.0f, 6.0f}));
      auto& y_values = *deleter.weights.back();
      auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

      // if you use this API the initializer data MUST remain valid for the lifetime of the InferenceSession
      Ort::ThrowOnError(
          api.CreateTensorWithDataAndDeleterAsOrtValue(&deleter,
                                                       y_values.data(), y_values.size() * sizeof(y_values[0]),
                                                       y_dims.data(), y_dims.size(),
                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                       &y_tensor));

      Ort::ThrowOnError(graph_api.AddInitializerToGraph(graph, "Y", y_tensor, /*data is external*/ true));
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

  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  std::vector<int64_t> expected_dims = {3, 3};
  ModelBuilderAPI::Model cxx_model(model);
  auto session = CreateSession(*ort_env, cxx_model);

  TestInference<float>(session, inputs, "Z", expected_dims,
                       {18.0f, 24.0f, 30.0f,
                        38.0f, 52.0f, 66.0f,
                        58.0f, 80.0f, 102.0f});

  api.ReleaseSession(session.release());

  ASSERT_EQ(deleter.weights.size(), 0) << "All weights should have been freed";
}

TEST(ModelBuilderAPITest, Basic_CxxApi) {
  // initializers that are used directly by the model. as there's no copy they must remain valid
  std::vector<std::unique_ptr<std::vector<float>>> weights;

  Ort::ModelBuilderAPI::Graph graph;

  //
  // Create OrtModel with a Gemm. X input is 3x2, Y input is 2x3, Z output is 3x3.
  // X is model input. Y is initializer.
  // Set the alpha attribute of the Gemm node to 2.0 to test attribute handling.
  //

  std::vector<ModelBuilderAPI::ValueInfo> graph_inputs;
  std::vector<ModelBuilderAPI::ValueInfo> graph_outputs;

  // model input. it's {3, 2} but use a symbolic dim to test that works.
  std::vector<int64_t> input_dims({-1, 2});
  std::vector<std::string> input_symbolic_dims({"multiple_of_3", ""});
  TensorTypeAndShapeInfo input_tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                           input_dims,
                                           &input_symbolic_dims);
  auto input_type_info = TypeInfo::CreateTensorInfo(input_tensor_info.GetConst());
  graph_inputs.emplace_back("X", input_type_info.GetConst());

  // model outputs
  std::vector<int64_t> output_dims = {-1, 3};
  std::vector<std::string> output_symbolic_dims({"multiple_of_3", ""});
  TensorTypeAndShapeInfo output_tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                            output_dims,
                                            &output_symbolic_dims);
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

  ModelBuilderAPI::Node node("Gemm", onnxruntime::kOnnxDomain, "Gemm1", {"X", "Y"}, {"Z"}, attributes);

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
  graph.AddInitializer("Y", y_tensor, /*data is external*/ true);

  std::vector<ModelBuilderAPI::Model::DomainOpsetPair> opsets{{onnxruntime::kOnnxDomain, 18}};
  ModelBuilderAPI::Model model(opsets);
  model.AddGraph(graph);

  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  std::vector<int64_t> expected_dims = {3, 3};

  auto session = CreateSession(*ort_env, model);
  TestInference<float>(session, inputs, "Z", expected_dims,
                       {18.0f, 24.0f, 30.0f,
                        38.0f, 52.0f, 66.0f,
                        58.0f, 80.0f, 102.0f});
}

TEST(ModelBuilderAPITest, BasicModelEdit_CxxApi) {
  //
  // Load existing model
  // Add Cast to change the model input from float to int64
  // Update model inputs to match
  // Run
  //

  SessionOptions so;

  // Set this to save the model if you want to debug.
  // so.SetOptimizedModelFilePath(ORT_TSTR("model_builder_edited.onnx"));

  Session session = Session::CreateModelBuilderSession(*ort_env, TSTR("testdata/mnist.onnx"), so);

  ASSERT_EQ(session.GetOpset(""), 8);  // ONNX domain is empty string

  // we augment the original model with nodes, initializers and the updated model inputs/outputs from this model.
  // the original graph is unchanged. nodes can be added before/after it. initializers can be added.
  // new nodes must conform to the original domain:opset of the model.
  // additional operator domain:opset pairs can be added.
  std::vector<ModelBuilderAPI::Model::DomainOpsetPair> opsets;  // no additional opsets required
  ModelBuilderAPI::Model model(opsets);

  std::vector<std::string> input_names = session.GetInputNames();
  ASSERT_EQ(input_names.size(), 1);

  TypeInfo orig_input = session.GetInputTypeInfo(0);
  ASSERT_EQ(orig_input.GetTensorTypeAndShapeInfo().GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  const std::string new_input_name = "Int64Input";

  // Add Cast node to convert input from float to int64
  std::vector<OpAttr> attributes;
  int64_t to = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  attributes.push_back(OpAttr("to", &to, 1, OrtOpAttrType::ORT_OP_ATTR_INT));

  ModelBuilderAPI::Node node("Cast", onnxruntime::kOnnxDomain, new_input_name, {"Int64Input"}, {input_names[0]},
                             attributes);

  // we're replacing the only input, so we don't need to call session.GetInputTypeInfo(x) to copy other inputs
  // in order to preserve them
  std::vector<ModelBuilderAPI::ValueInfo> graph_inputs;
  TensorTypeAndShapeInfo input_tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                           orig_input.GetTensorTypeAndShapeInfo().GetShape());
  auto input_type_info = TypeInfo::CreateTensorInfo(input_tensor_info.GetConst());
  graph_inputs.emplace_back(new_input_name, input_type_info.GetConst());

  ModelBuilderAPI::Graph graph;  // new info to augment the model with

  graph.AddNode(node);
  graph.SetInputs(graph_inputs);

  // the node we added does not require any new opsets.
  model.AddGraph(graph);

  session.FinalizeModelBuilderSession(model, so);

  std::vector<Input<int64_t>> inputs(1);
  auto& input = inputs[0];
  input.name = new_input_name.c_str();
  input.dims = orig_input.GetTensorTypeAndShapeInfo().GetShape();

  auto num_values = std::accumulate(input.dims.begin(), input.dims.end(), int64_t(1), std::multiplies<int64_t>());
  input.values.resize(size_t(num_values));
  std::iota(input.values.begin(), input.values.end(), 1);

  std::vector<int64_t> expected_dims = {1, 10};
  std::vector<float> expected_output = {-48.5088f, -1040.2948f, -347.0959f, 101.7392f, 421.3352f,
                                        750.92145f, 231.5060f, -1694.4152f, 681.5623f, 378.1689f};

  TestInference<float>(session, inputs, session.GetOutputNames()[0].c_str(), expected_dims, expected_output);

  // double check with original model
  {
    SessionOptions expected_so;
    Session expected_session = Session(*ort_env, TSTR("testdata/mnist.onnx"), expected_so);
    std::vector<Input<float>> expected_inputs(1);
    auto& expected_input = expected_inputs[0];
    expected_input.name = input_names[0].c_str();
    expected_input.dims = orig_input.GetTensorTypeAndShapeInfo().GetShape();
    expected_input.values.reserve(size_t(num_values));
    std::transform(input.values.begin(), input.values.end(), std::back_inserter(expected_input.values),
                   [&](int64_t value) { return float(value); });

    TestInference<float>(expected_session, expected_inputs, session.GetOutputNames()[0].c_str(),
                         expected_dims, expected_output);
  }
}

/*
Tests required

- Constant node is converted to initializer
- Attempt to create invalid model
- Edit and change outputs
- Invalid edit
- Edit where we change a subset of inputs or outputs.
*/
