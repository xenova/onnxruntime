// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <variant>

#include "core/framework/error_code_helper.h"
#include "core/framework/ort_value.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/graph_apis.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"

struct OrtShape {
  ONNX_NAMESPACE::TensorShapeProto shape_proto;
};

struct OrtValueInfo {
  ONNX_NAMESPACE::ValueInfoProto value_info_proto;
};

struct OrtOpAttr {
  ONNX_NAMESPACE::AttributeProto attr_proto;
};

struct OrtNode {
  std::string operator_name;
  std::string domain_name;
  std::string node_name;

  // OrtOpAttr is 1:1 with ONNX_NAMESPACE::AttributeProto currently.
  // https://github.com/microsoft/onnxruntime/blob/bd5a759d0cdbed6e7f611c990d4eb5457a9ecf60/onnxruntime/core/session/standalone_op_invoker.cc#L318
  // Might be better if it had a wrapper struct so we have more flexibility.
  // AFAIK (TBC) that's an implementation detail so we should be able to change it.
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // FUTURE if we need control flow nodes
  // std::unordered_map<std::string, OrtGraph> subgraphs;
};

struct OrtGraph {
  std::vector<std::unique_ptr<OrtValueInfo>> inputs;
  std::vector<std::unique_ptr<OrtValueInfo>> outputs;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> initializers;
  std::vector<std::unique_ptr<OrtNode>> nodes;
};

struct OrtModel {
  std::unique_ptr<OrtGraph> graph;
  std::unordered_map<std::string, int64_t> domain_to_version;
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
                    _In_ OrtShape* shape, _Outptr_ OrtValueInfo** value_info) {
  API_IMPL_BEGIN
  auto vi = std::make_unique<OrtValueInfo>();
  vi->value_info_proto.set_name(name);
  auto* tensor = vi->value_info_proto.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(type);
  *tensor->mutable_shape() = shape->shape_proto;

  *value_info = vi.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseValueInfo, _Frees_ptr_opt_ OrtValueInfo* value_info) {
  delete value_info;
}

ORT_API_STATUS_IMPL(CreateNode, const char* operator_name, const char* domain_name, _In_ const char* node_name,
                    _In_reads_(input_names_len) const char* const* input_names, size_t input_names_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _In_reads_(attribs_len) _In_opt_ const OrtOpAttr* const* attributes, _In_opt_ size_t attribs_len,
                    _Outptr_ OrtNode** node) {
  API_IMPL_BEGIN
  auto n = std::make_unique<OrtNode>();
  n->operator_name = operator_name;
  n->domain_name = domain_name;
  n->node_name = node_name;

  n->input_names.reserve(input_names_len);
  for (size_t i = 0; i < input_names_len; ++i) {
    n->input_names.push_back(input_names[i]);
  }

  n->output_names.reserve(output_names_len);
  for (size_t i = 0; i < output_names_len; ++i) {
    n->output_names.push_back(output_names[i]);
  }

  if (attributes != nullptr) {
    n->attributes.reserve(attribs_len);
    for (size_t i = 0; i < attribs_len; ++i) {
      n->attributes.push_back(*reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(attributes[i]));
    }
  }

  *node = n.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseNode, _Frees_ptr_opt_ OrtNode* node) {
  delete node;
}

ORT_API_STATUS_IMPL(CreateGraph, _Outptr_ OrtGraph** graph) {
  API_IMPL_BEGIN
  auto g = std::make_unique<OrtGraph>();

  // do some reserves to reduce reallocation. if we had a hint about sizes upfront that would be optimal
  g->inputs.reserve(8);
  g->outputs.reserve(8);
  g->initializers.reserve(64);
  g->nodes.reserve(64);

  *graph = g.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(AddInput, _In_ OrtGraph* graph, _Inout_ OrtValueInfo** value_info) {
  API_IMPL_BEGIN
  graph->inputs.push_back(std::unique_ptr<OrtValueInfo>(*value_info));  // take ownership
  *value_info = nullptr;
  return nullptr;
  API_IMPL_END
}
ORT_API_STATUS_IMPL(AddOutput, _In_ OrtGraph* graph, _Inout_ OrtValueInfo** value_info) {
  API_IMPL_BEGIN
  graph->outputs.push_back(std::unique_ptr<OrtValueInfo>(*value_info));  // take ownership
  *value_info = nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(AddInitializer, _In_ OrtGraph* graph, _In_ const char* name, _Inout_ OrtValue** tensor) {
  API_IMPL_BEGIN
  graph->initializers[name] = std::unique_ptr<OrtValue>(*tensor);  // take ownership
  *tensor = nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(AddNode, _In_ OrtGraph* graph, _Inout_ OrtNode** node) {
  API_IMPL_BEGIN
  graph->nodes.push_back(std::unique_ptr<OrtNode>(*node));  // take ownership
  *node = nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseGraph, _Frees_ptr_opt_ OrtGraph* graph) {
  delete graph;
}

ORT_API_STATUS_IMPL(CreateModel,
                    _In_reads_(opset_entries_len) const char* const* domain_names,
                    _In_reads_(opset_entries_len) const size_t* const* opset_versions,
                    size_t opset_entries_len,
                    _Outptr_ OrtModel** model) {
  API_IMPL_BEGIN
  auto m = std::make_unique<OrtModel>();
  for (size_t i = 0; i < opset_entries_len; ++i) {
    m->domain_to_version[domain_names[i]] = *opset_versions[i];
  }

  *model = m.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(AddGraph, _In_ OrtModel* model, _Inout_ OrtGraph** graph) {
  API_IMPL_BEGIN
  model->graph = std::unique_ptr<OrtGraph>(*graph);  // take ownership
  *graph = nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseModel, _Frees_ptr_opt_ OrtModel* model) {
  delete model;
}

ORT_API_STATUS_IMPL(CreateSessionFromModel, _In_ const OrtEnv* /*env*/, _Inout_ OrtModel** /*model*/,
                    _In_ const OrtSessionOptions* /*options*/, _Outptr_ OrtSession** /*out*/) {
  // TODO: Add InferenceSession ctor that takes OrtModel.
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "CreateSessionFromModel is not implemented");
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

    &OrtGraphApis::CreateNode,
    &OrtGraphApis::ReleaseNode,

    &OrtGraphApis::CreateGraph,
    &OrtGraphApis::AddInput,
    &OrtGraphApis::AddOutput,
    &OrtGraphApis::AddInitializer,
    &OrtGraphApis::AddNode,
    &OrtGraphApis::ReleaseGraph,

    &OrtGraphApis::CreateModel,
    &OrtGraphApis::AddGraph,
    &OrtGraphApis::ReleaseModel,

    &OrtGraphApis::CreateSessionFromModel,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtGraphApi, CreateSessionFromModel) / sizeof(void*) == 18,
              "Size of version 21 API cannot change");  // initial version in ORT 1.21

ORT_API(const OrtGraphApi*, OrtGraphApis::GetGraphApi) {
  // No constraints on the API version yet.
  return &ort_graph_api;
}
