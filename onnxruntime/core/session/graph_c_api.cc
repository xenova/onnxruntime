// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <variant>

#include "core/framework/error_code_helper.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/graph_apis.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"

using Dimension = std::variant<int64_t, std::string>;
struct OrtShape {
  ONNX_NAMESPACE::TensorShapeProto shape_proto;
};

struct OrtValueInfo {
  ONNX_NAMESPACE::ValueInfoProto value_info_proto;
};

ORT_API_STATUS_IMPL(CreateModel,
                    _In_reads_(opset_entries_len) const char* const* domain_names,
                    _In_reads_(opset_entries_len) const size_t* const* opset_versions,
                    size_t opset_entries_len,
                    _Outptr_ OrtModel** model) {

}

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

ORT_API_STATUS_IMPL(AddNode, _In_ OrtGraph* graph, _In_ const char* op_type, _In_ const char* op_name,
                    _In_reads_(input_names_len) const char* const* input_names, size_t input_names_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _In_reads_(attribs_len) _In_opt_ const OrtOpAttr* const* attributes, _In_opt_ size_t attribs_len,
                    _Outptr_ OrtNode** node) {
}

// Design choice: Should we require a Model to be created first with the Graph to come from that so that we can use
// the C++ types directly?
ORT_API_STATUS_IMPL(CreateGraph, _Outptr_ OrtGraph** graph) {
  API_IMPL_BEGIN

  // *graph = new OrtGraph();
  return nullptr;
  API_IMPL_END
}

/*
ORT_API_STATUS_IMPL(GetGraph, _In_ OrtModel* model, _Outptr_ OrtGraph** graph);
ORT_API_STATUS_IMPL(CreateSubGraph, _Outptr_ OrtGraph** graph);

ORT_API_STATUS_IMPL(AddInput, _In_ OrtGraph* graph, _In_ const OrtValueInfo* value_info);
ORT_API_STATUS_IMPL(AddOutput, _In_ OrtGraph* graph, _In_ const OrtValueInfo* value_info);
ORT_API_STATUS_IMPL(AddInitializer, _In_ OrtGraph* graph, _In_ const char* name, _In_ OrtValue* tensor);

ORT_API_STATUS_IMPL(CreateModel,
                  _In_reads_(opset_entries_len) const char* const* domain_names,
                  _In_reads_(opset_entries_len) const size_t* const* opset_versions,
                  size_t opset_entries_len,
_Outptr_ OrtModel** model);
ORT_API_STATUS_IMPL(AddOpsetImport, _In_ OrtModel* model, _In_ const char* domain, _In_ int64_t version);
ORT_API_STATUS_IMPL(AddGraph, _In_ OrtModel* model, _In_ OrtGraph* graph);
ORT_API(void, ReleaseModel, _Frees_ptr_opt_ OrtModel*);

ORT_API_STATUS_IMPL(CreateSessionFromModel, _In_ const OrtEnv* env, _In_ OrtModel* model,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);
*/
static constexpr OrtGraphApi ort_graph_api = {
    // NOTE: The C# bindings depend on the API order within this struct so all additions must be at the end,
    // and no functions can be removed (the implementation needs to change to return an error).
    &OrtGraphApis::CreateModel,
    &OrtGraphApis::AddGraph,
    &OrtGraphApis::ReleaseModel,

    &OrtGraphApis::CreateGraph,
    &OrtGraphApis::GetGraph,
    &OrtGraphApis::CreateSubGraph,
    &OrtGraphApis::AddInput,
    &OrtGraphApis::AddOutput,
    &OrtGraphApis::AddInitializer,
    &OrtGraphApis::AddNode,

    &OrtGraphApis::CreateFixedShape,
    &OrtGraphApis::CreateShape,
    &OrtGraphApis::AddDimension,
    &OrtGraphApis::AddDynamicDimension,
    &OrtGraphApis::ReleaseShape,

    &OrtGraphApis::CreateTensorValueInfo,
    &OrtGraphApis::ReleaseValueInfo,

    &OrtGraphApis::CreateSessionFromModel,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtGraphApi, CreateSessionFromModel) / sizeof(void*) == 17, "Size of version 21 API cannot change");

ORT_API(const OrtGraphApi*, OrtGraphApis::GetGraphApi) {
  // No constraints on the API version yet.
  return &ort_graph_api;
}
