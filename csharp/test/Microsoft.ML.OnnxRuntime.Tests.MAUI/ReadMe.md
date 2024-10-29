The MAUI test project can be optionally used with a pre-built ONNX Runtime native nuget package (Microsoft.ML.OnnxRuntime).

To do so, specify the `UsePrebuiltNativePackage` and `CurrentOnnxRuntimeVersion` properties when building the project.
These can be set via the command-line or as environment variables.

For example:

```cmd
dotnet build csharp\test\Microsoft.ML.OnnxRuntime.Tests.MAUI\Microsoft.ML.OnnxRuntime.Tests.MAUI.csproj --property:UsePrebuiltNativePackage=true --property:CurrentOnnxRuntimeVersion=1.19.2 --source directory_containing_native_nuget_package --source https://api.nuget.org/v3/index.json
```

**Important note**
The OrtApi struct defined in
[onnxruntime_c_api.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/session/onnxruntime_c_api.cc)
in the pre-built native package that is returned by the GetApi call must be compatible with the OrtApi struct defined in
[NativeMethods.shared.cs](https://github.com/microsoft/onnxruntime/blob/main/csharp/src/Microsoft.ML.OnnxRuntime/NativeMethods.shared.cs)
in the C# managed code. If it is not there will be a failure at startup from attempting to call
GetDelegateForFunctionPointer for a non-existent function.

Functions are always appended to both OrtApi structs, so the last function name defined in the C# struct must
exist in the native OrtApi struct. It is no problem for the native library to have additional functions as they will be
ignored.
