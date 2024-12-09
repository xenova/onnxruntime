// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <variant>

#include "core/framework/error_code_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/to_tensor_proto_element_type.h"
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/graph_apis.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"

using namespace onnxruntime;
using Dimension = std::variant<int64_t, std::string>;

struct OrtShape {
  ONNX_NAMESPACE::TensorShapeProto shape_proto;
};

struct OrtValueInfo {
  ONNX_NAMESPACE::ValueInfoProto value_info_proto;
};

struct OrtOpAttr {
  ONNX_NAMESPACE::AttributeProto attr_proto;
};

struct OrtSession {
  std::unique_ptr<onnxruntime::InferenceSession> instance;
};

struct OrtModel {
  onnxruntime::Model& model;  // model from OrtSession.instance
};

struct OrtGraph {
  onnxruntime::Graph& graph;  // Graph from OrtModel.model
};

struct OrtNode {
  onnxruntime::Node& node;  // Node from OrtGraph.graph
};

ORT_API_STATUS_IMPL(CreateShape, _Outptr_ OrtShape** shape) {
  API_IMPL_BEGIN
  *shape = new OrtShape();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(AddDimension, _In_ OrtShape* shape, int64_t dim_value) {
  API_IMPL_BEGIN
  shape->shape_proto.add_dim()->set_dim_value(dim_value);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(AddDynamicDimension, _In_ OrtShape* shape, const char* dimension_name) {
  API_IMPL_BEGIN
  if (dimension_name == nullptr || dimension_name == '\0') {
    shape->shape_proto.add_dim();  // 'unknown'dimension exists but has neither dim_value nor dim_param
  } else {
    shape->shape_proto.add_dim()->set_dim_param(dimension_name);
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(CreateFixedShape, _In_ const int64_t* dim_values, size_t dim_count, _Outptr_ OrtShape** shape) {
  API_IMPL_BEGIN
  auto s = std::make_unique<OrtShape>();
  for (size_t i = 0; i < dim_count; ++i) {
    s->shape_proto.add_dim()->set_dim_value(dim_values[i]);
  }

  *shape = s.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseShape, _Frees_ptr_opt_ OrtShape* shape) {
  delete shape;
}

ORT_API_STATUS_IMPL(CreateTensorValueInfo, _In_ const char* name, _In_ ONNXTensorElementDataType type,
                    _Inout_ OrtShape** shape, _Outptr_ OrtValueInfo** value_info) {
  API_IMPL_BEGIN
  auto vi = std::make_unique<OrtValueInfo>();
  vi->value_info_proto.set_name(name);
  auto* tensor = vi->value_info_proto.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(type);
  *tensor->mutable_shape() = (*shape)->shape_proto;

  delete *shape;  // take ownership
  *shape = nullptr;

  *value_info = vi.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseValueInfo, _Frees_ptr_opt_ OrtValueInfo* value_info) {
  delete value_info;
}

ORT_API_STATUS_IMPL(CreateSession, _In_ const OrtEnv* env, _In_ const OrtSessionOptions* options,
                    _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  OrtSession session;
  session.instance = std::make_unique<onnxruntime::InferenceSession>(options->value,
                                                                     env->GetEnvironment());
  *out = new OrtSession(std::move(session));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(FinalizeModel, _In_ OrtSession* /*session*/) {
  // TODO: Add method to InferenceSession and Model to resolve the Graph and run InferenceSession::Initialize
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "FinalizeModel is not implemented");
}

ORT_API_STATUS_IMPL(CreateModel,
                    _In_reads_(opset_entries_len) const char* const* /*domain_names*/,
                    _In_reads_(opset_entries_len) const int* const* /*opset_versions*/,
                    size_t /*opset_entries_len*/,
                    _Outptr_ OrtModel** model) {
  // TODO: InferenceSession needs to provide a method to construct the empty Model instance and return it.
  // When it creates the Model instance it should plugin the session logger and any custom op registries.
  // Might need a static method on onnxruntime::Model like we use for LoadFromOrtFormat.
  *model = nullptr;
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "CreateModel is not implemented");
}

ORT_API_STATUS_IMPL(GetGraph, _In_ OrtModel* model, _Outptr_ OrtGraph** graph) {
  API_IMPL_BEGIN
  *graph = new OrtGraph{model->model.MainGraph()};
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(AddInput, _In_ OrtGraph* graph, _Inout_ OrtValueInfo** value_info) {
  API_IMPL_BEGIN
  auto& node_arg = graph->graph.GetOrCreateNodeArg((*value_info)->value_info_proto.name(),
                                                   &(*value_info)->value_info_proto.type());

  std::vector<const NodeArg*> inputs = graph->graph.GetInputs();  // copy
  inputs.push_back(&node_arg);
  graph->graph.SetInputs(inputs);

  delete *value_info;  // take ownership
  *value_info = nullptr;
  return nullptr;
  API_IMPL_END
}
ORT_API_STATUS_IMPL(AddOutput, _In_ OrtGraph* graph, _Inout_ OrtValueInfo** value_info) {
  API_IMPL_BEGIN
  auto& node_arg = graph->graph.GetOrCreateNodeArg((*value_info)->value_info_proto.name(),
                                                   &(*value_info)->value_info_proto.type());

  std::vector<const NodeArg*> cur_inputs = graph->graph.GetOutputs();  // copy
  cur_inputs.push_back(&node_arg);
  graph->graph.SetOutputs({&node_arg});

  delete *value_info;  // take ownership
  *value_info = nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(AddInitializer, _In_ OrtGraph* graph, _In_ const char* name, _Inout_ OrtValue** value) {
  API_IMPL_BEGIN
  OrtValue& v = **value;
  ORT_ENFORCE(v.IsTensor());
  const Tensor& t = v.Get<Tensor>();

  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(t.GetElementType());
  for (auto dim : t.Shape().GetDims()) {
    tensor_proto.add_dims(dim);
  }

  // TODO: Infer if CreateTensorWithDataAsOrtValue or CreateTensorAsOrtValue was used based on whether the
  // Tensor in the OrtValue owns the buffer.
  const bool is_internal_data = t.OwnsBuffer();

  if (is_internal_data) {
    tensor_proto.set_raw_data(t.DataRaw(), t.SizeInBytes());
  } else {
    tensor_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

    const void* data_offset = t.DataRaw();  // actual address of memory not offset into file
    auto offset = narrow<ExternalDataInfo::OFFSET_TYPE>(reinterpret_cast<intptr_t>(data_offset));

    ONNX_NAMESPACE::StringStringEntryProto* entry = tensor_proto.mutable_external_data()->Add();
    entry->set_key("location");
    // magic tag for existing memory that causes 'offset' to be treated as a pointer to the memory
    entry->set_value(ToUTF8String(onnxruntime::utils::kTensorProtoMemoryAddressTag));
    entry = tensor_proto.mutable_external_data()->Add();
    entry->set_key("offset");
    entry->set_value(std::to_string(offset));
    entry = tensor_proto.mutable_external_data()->Add();
    entry->set_key("length");
    entry->set_value(std::to_string(t.SizeInBytes()));
  }

  graph->graph.AddInitializedTensor(tensor_proto);  // note: this copies the TensorProto

  delete *value;  // take ownership
  *value = nullptr;

  return nullptr;
  API_IMPL_END
}
ORT_API_STATUS_IMPL(AddNode, _In_ OrtGraph* graph,
                    _In_ const char* operator_name, const char* domain_name, _In_ const char* node_name,
                    _In_reads_(input_names_len) const char* const* input_names, size_t input_names_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _In_reads_(attribs_len) _In_opt_ OrtOpAttr** attributes, _In_opt_ size_t attribs_len,
                    _Outptr_ OrtNode** out) {
  API_IMPL_BEGIN
  std::vector<NodeArg*> inputs;
  inputs.reserve(input_names_len);
  for (size_t i = 0; i < input_names_len; ++i) {
    // type info will be inferred.
    inputs.push_back(&graph->graph.GetOrCreateNodeArg(input_names[i], /*TypeProto*/ nullptr));
  }

  std::vector<NodeArg*> outputs;
  outputs.reserve(output_names_len);
  for (size_t i = 0; i < output_names_len; ++i) {
    outputs.push_back(&graph->graph.GetOrCreateNodeArg(output_names[i], /*TypeProto*/ nullptr));
  }

  NodeAttributes node_attrs;
  if (attribs_len) {
    node_attrs.reserve(attribs_len);
    for (size_t i = 0; i < attribs_len; ++i) {
      node_attrs[attributes[i]->attr_proto.name()] = attributes[i]->attr_proto;

      delete attributes[i];  // take ownership
      attributes[i] = nullptr;
    }
  }

  // default domain to onnx (empty string)
  const auto* domain = domain_name ? domain_name : "";

  auto& node = graph->graph.AddNode(node_name, operator_name, /*description*/ "", inputs, outputs, &node_attrs,
                                    domain);

  *out = new OrtNode{node};
  return nullptr;
  API_IMPL_END
}

static constexpr OrtGraphApi ort_graph_api = {
    // NOTE: The C# bindings depend on the API order within this struct so all additions must be at the end,
    // and no functions can be removed (the implementation needs to change to return an error).
    &OrtGraphApis::CreateFixedShape,
    &OrtGraphApis::CreateShape,
    &OrtGraphApis::AddDimension,
    &OrtGraphApis::AddDynamicDimension,
    &OrtGraphApis::ReleaseShape,

    &OrtGraphApis::CreateTensorValueInfo,
    &OrtGraphApis::ReleaseValueInfo,

    &OrtGraphApis::CreateSession,
    &OrtGraphApis::FinalizeModel,

    &OrtGraphApis::CreateModel,

    &OrtGraphApis::GetGraph,
    &OrtGraphApis::AddInput,
    &OrtGraphApis::AddOutput,
    &OrtGraphApis::AddInitializer,
    &OrtGraphApis::AddNode,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtGraphApi, AddNode) / sizeof(void*) == 14,
              "Size of version 21 API cannot change");  // original version in ORT 1.21

ORT_API(const OrtGraphApi*, OrtGraphApis::GetGraphApi) {
  // No constraints on the API version yet.
  return &ort_graph_api;
}
