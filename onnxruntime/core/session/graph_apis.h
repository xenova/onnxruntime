// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace OrtGraphApis {

// implementation that returns the API struct
ORT_API(const OrtGraphApi*, GetGraphApi);

ORT_API_STATUS_IMPL(CreateFixedShape, _In_ const int64_t* dim_values, size_t dim_count, _Outptr_ OrtShape** shape);
ORT_API_STATUS_IMPL(CreateShape, _Outptr_ OrtShape** shape);
ORT_API_STATUS_IMPL(AddDimension, _In_ OrtShape* shape, int64_t dim_value);
ORT_API_STATUS_IMPL(AddDynamicDimension, _In_ OrtShape* shape, const char* dimension_name);
ORT_API(void, ReleaseShape, _Frees_ptr_opt_ OrtShape* shape);

ORT_API_STATUS_IMPL(CreateTensorValueInfo, _In_ const char* name, _In_ ONNXTensorElementDataType type,
                    _In_ OrtShape* shape, _Outptr_ OrtValueInfo** value_info);
ORT_API(void, ReleaseValueInfo, _Frees_ptr_opt_ OrtValueInfo* value_info);

ORT_API_STATUS_IMPL(CreateNode, const char* operator_name, const char* domain_name, _In_ const char* node_name,
                    _In_reads_(input_names_len) const char* const* input_names, size_t input_names_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _In_reads_(attribs_len) _In_opt_ const OrtOpAttr* const* attributes, _In_opt_ size_t attribs_len,
                    _Outptr_ OrtNode** node);
ORT_API(void, ReleaseNode, _Frees_ptr_opt_ OrtNode* node);

ORT_API_STATUS_IMPL(CreateGraph, _Outptr_ OrtGraph** graph);
ORT_API_STATUS_IMPL(AddInput, _In_ OrtGraph* graph, _Inout_ OrtValueInfo** value_info);
ORT_API_STATUS_IMPL(AddOutput, _In_ OrtGraph* graph, _Inout_ OrtValueInfo** value_info);
ORT_API_STATUS_IMPL(AddInitializer, _In_ OrtGraph* graph, _In_ const char* name, _Inout_ OrtValue** tensor);
ORT_API_STATUS_IMPL(AddNode, _In_ OrtGraph* graph, _Inout_ OrtNode** node);
ORT_API(void, ReleaseGraph, _Frees_ptr_opt_ OrtGraph* graph);

ORT_API_STATUS_IMPL(CreateModel,
                    _In_reads_(opset_entries_len) const char* const* domain_names,
                    _In_reads_(opset_entries_len) const size_t* const* opset_versions,
                    size_t opset_entries_len,
                    _Outptr_ OrtModel** model);
ORT_API_STATUS_IMPL(AddGraph, _In_ OrtModel* model, _Inout_ OrtGraph** graph);
ORT_API(void, ReleaseModel, _Frees_ptr_opt_ OrtModel* model);

ORT_API_STATUS_IMPL(CreateSessionFromModel, _In_ const OrtEnv* env, _Inout_ OrtModel** model,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);
}  // namespace OrtGraphApis
