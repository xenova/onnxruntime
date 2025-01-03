// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api_ep.h"

namespace Ort {
namespace PluginEP {

//// High level questions:

// what's the reason for not using the same approach as the ORT C++ API with a base class?
//  - cost/benefit of using a different approach?

// What's the minimum C++ version we need to support? There are C++17 types used here (e.g. std::filesystem::path).

// We need to figure out how/where we want to manage memory in general
//   - when to use std::string vs const char*
//     - e.g. GetAllInitializers allocating a std::string for every initializer could be a perf issue
//   - do we need to return a heap allocated value of the API structs in so many places?
//   - usage of shared_ptr vs. unique_ptr if we allocate a struct that the user takes ownership of
//     - prefer unique_ptr unless there's a specific need to use shared_ptr
//   - there are places where we replicate an internal ORT API (where returning a container class as a reference is
//     cheap/convenient) and return a new container with copies of the elements. do we need to do that vs. providing a
//     lookup function instead?
//     e.g. do we need GraphViewer::GetAllInitializers() or would GraphViewer::GetInitializer(const char* name) work?

// would love to merge Graph and GraphViewer into a single API class.
//
// we have 3 (at least) conceptual versions of a 'graph'
//
// 1. the main graph
// 2. a subgraph from a control flow node (If/Loop/Scan). essentially the same as #1 + outer scope values
// 3. a subgraph from a ComputeCapability::IndexedSubGraph which is a subset of nodes from #1 or #2. this differs in
//    we filter the visible nodes and initializers based on the IndexedSubGraph in the GraphViewer.
//
// A GraphViewer is used for all 3 of these things so from an EP author's perspective do they need to know there are
// 2 different 'graph' classes in ORT or can we just use a single class in the OOT API? the internal implementation
// could use either an onnxruntime::Graph or onnxruntime::GraphViewer (or both).
//

// lot more things should be `const`. that would give us more flexibility if we wanted to cache some things and
// return a const reference to it instead of always allocating a new struct when returning the value to the user.

// Need to clarify expected usage of OrtGraph_CreateOrUpdateEpCtxGraph.
//   Is the EP responsible for creating the entire Graph?
//   How does it work if there may be multiple EPs?
// Naively I would have expected we'd handle this more directly as part of creating the compiled node vs. doing it
// as a step at the end of partitioning.
// What is the reason for delaying until the end and recreating the whole Graph?
// If we can't do this more directly when creating the compiled node, can we structure it more like the current
// implementation where ORT is in charge of creating the Graph and asks the EPs for their EP context nodes?

// Is pre-packing supported? Do we need to add that to the custom op API?

// using 'Ptr' as a general suffix for a unique_ptr is not overly clear as someone reading the code needs to go find
// the definition of the type to know how to treat it correctly. can we use 'UniquePtr' as the suffix instead?
using VoidPtr = std::unique_ptr<void, std::function<void(void*)>>;

// Can we use OrtValue/Ort::Value and the TypeInfo* structs (OrtTypeInfo/TensorTypeAndShapeInfo) instead or TensorRef
// and ValueInfoRef?
// May need to expand that API a little to make it easier to discover the type (tensor/sparse tensor/sequence/map/unique)
// but a lot of the functionality is already there and using it would make things more consistent overall as well as
// supporting all data types.
// Caveat: If we add quantization into to ValueInfo we obviously need a different type here
struct TensorRef {
  explicit TensorRef(OrtTensorRef*);
  ~TensorRef();
  const std::vector<int64_t> GetShape();
  const ONNXTensorElementDataType GetTensorElementType();
  const char* GetData();  // why const char* and not void*?
  size_t GetDataLen();

 private:
  OrtTensorRef* tensor_;
};

struct ValueInfoRef {
  explicit ValueInfoRef(OrtValueInfoRef*);
  ~ValueInfoRef();
  const std::vector<int64_t> GetShape();
  const ONNXTensorElementDataType GetTensorElementType();

 private:
  OrtValueInfoRef* value_info_;
};

struct Graph {
  explicit Graph(const OrtGraph*);
  const OrtGraph* GetGraph() { return graph_; }
  // this creates a requirement for C++ 17. what minimum C++ version do we need to support?
  void DumpOnnxModel(const std::filesystem::path& onnx_model_path);

 private:
  const OrtGraph* graph_;
};
using GraphPtr = std::unique_ptr<PluginEP::Graph, std::function<void(PluginEP::Graph*)>>;

struct GraphViewer {
  explicit GraphViewer(const OrtGraphViewer*);
  // do we need this if the C++ API is comprehensive?
  // or should we have an `operator OrtGraphViewer*()` to be consistent with how the ORT C++ API behaves?
  const OrtGraphViewer* GetGraphViewer() { return graph_; }
  const char* GetName();
  bool IsConstantInitializer(const char* name, bool check_outer_scope);
  const std::vector<size_t> GetNodeIndexesInTopologicalOrder(int execution_order);  // do we need an execution order arg?
  bool IsSubgraph();
  std::shared_ptr<Node> GetParentNode();
  std::filesystem::path GetModelPath();
  std::vector<std::string> GetRequiredInputs();
  std::vector<std::string> GetAllInputs();
  // this seems expensive. do we need all at once or just a way to do lookup of a name to check if it's an initializer?
  std::vector<std::string> GetAllInitializers();
  Node GetOrtNode(size_t node_index);  // GetNode?
  std::vector<Node> GetNodesConsumingInput(const char* input_name);
  Node GetNodeProducingOutput(const char* output_name);
  int NumberOfNodes();  // Does an EP need to know how many nodes there are?
  int MaxNodeIndex();
  // Could we replace these with something like std::vector<const char*> GetOutputs() const?
  size_t GetOutputSize();
  std::string GetIthOutputName(size_t i);

  // this feels overly specific as the output could be a sequence/map/optional. Should we return OrtTypeInfo instead?
  // alternatively we could move the OrtValueInfo type in the new Model Builder API (https://github.com/microsoft/onnxruntime/pull/23223)
  // up to the ORT API and use it for the inputs/outputs
  int32_t GetIthOutputElemType(size_t i);

  std::shared_ptr<TensorRef> GetInitializerTensor(const char* initializer_name);  // GetInitializer?
  std::shared_ptr<ValueInfoRef> GetValueInfo(const char* name);

  std::pair<VoidPtr, size_t> SerializeToArray();

  // this feels like it should be something we control vs. the user calling it.
  // e.g. GraphPartitioner calls Compile and if applicable calls something to create/update the EP context graph.
  GraphPtr CreateOrUpdateEpCtxGraph(const char* node_name,
                                    const int64_t main_context,
                                    const int64_t embed_mode,
                                    const char* cache_path,
                                    char* cache_data,
                                    size_t size,
                                    const char* const* extra_attr_keys,
                                    const char* const* extra_attr_values,
                                    size_t extra_attr_num);

  // what's the use case for this?
  GraphViewerPtr GetSubGraph(std::vector<size_t> node_indices);

  // would operator== would be more typical C++ usage?
  bool IsSameGraph(GraphViewer& other);

 private:
  const OrtGraphViewer* graph_;
};
using GraphViewerPtr = std::unique_ptr<PluginEP::GraphViewer, std::function<void(PluginEP::GraphViewer*)>>;

struct Node {
  explicit Node(const OrtNode*);
  const char* GetName();  // inconsistent to return const char* for this and std::string for the other string getters
  const std::string GetDescription();
  const std::string GetDomain();
  int SinceVersion();
  const std::string GetExecutionProviderType();
  const std::string GetOpType();
  size_t GetIndex();

  // having to get the size followed by individual names feels a little painful for a C++ API vs. something like
  // std::vector<const char*> GetInputs().
  size_t GetNumInputs();
  const std::string GetIthInputName(size_t i);
  size_t GetImplicitInputSize();
  const std::string GetIthImplicitInputName(size_t i);
  size_t GetNumOutputs();
  const std::string GetIthOutputName(size_t i);

  std::vector<std::string> GetAttributeNames();
  size_t GetAttributeSize();
  // these could take `const std::string& attribute_name` and be const
  int GetAttributeType(std::string attribute_name);
  size_t GetAttributeKeyCount(std::string attribute_name);
  // can we have GetAttributeFloats(std::string attribute_name, size_t& size, const float* values) and
  // similar for int and string so a single call can be used?
  int GetAttributeIntSize(std::string attribute_name);
  int GetAttributeFloatSize(std::string attribute_name);
  int GetAttributeStringSize(std::string attribute_name);
  int64_t GetAttributeIthInt(std::string attribute_name, size_t i);
  float GetAttributeIthFloat(std::string attribute_name, size_t i);
  const std::string GetAttributeIthStr(std::string attribute_name, size_t i);
  const std::string GetAttributeStr(std::string attribute_name);
  int64_t GetAttributeInt(std::string attribute_name);
  float GetAttributeFloat(std::string attribute_name);
  // TODO: add GetSubgraphs wrapper here
 private:
  const OrtNode* node_;
};

}  // namespace PluginEP
}  // namespace Ort

#include "onnxruntime_cxx_inline_ep.h"
