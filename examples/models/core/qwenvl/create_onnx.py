# improved_convert_onnx.py
import onnx
from onnx import external_data_helper
import os

model_path = '/home/ljc/llm_deploy_proj/trt_llm_framework/TensorRT-LLM-main/model_zoo_trt_llm/qwen_vl_onnx/visual_encoder.onnx'

print("Loading ONNX model...")
model = onnx.load(model_path)

# 检查转换前的状态
print("=== Before conversion ===")
initial_external_count = 0
for initializer in model.graph.initializer:
    if initializer.HasField('data_location') and initializer.data_location == onnx.TensorProto.EXTERNAL:
        initial_external_count += 1

print(f"Number of tensors with external data: {initial_external_count}")

# 如果有外部数据，则进行转换
if initial_external_count > 0:
    print("Converting external data to internal data...")
    # convert_model_to_external_data(model, False) 的第二个参数为False表示不使用外部数据
    # 但我们实际需要的是将现有外部数据内联，所以使用load_external_data_for_model
    onnx.load_external_data_for_model(model, os.path.dirname(model_path))
    
    # 清除外部数据引用
    for initializer in model.graph.initializer:
        if initializer.HasField('data_location') and initializer.data_location == onnx.TensorProto.EXTERNAL:
            initializer.ClearField('data_location')
            initializer.ClearField('external_data')
    
    print("Saving model with internal data...")
    onnx.save(model, model_path)
    print("External data converted to internal data successfully!")
else:
    print("No external data found. Model already has all internal data.")

# 验证转换结果
print("\n=== After conversion ===")
model = onnx.load(model_path)
final_external_count = 0
for initializer in model.graph.initializer:
    if initializer.HasField('data_location') and initializer.data_location == onnx.TensorProto.EXTERNAL:
        final_external_count += 1

print(f"Number of tensors with external data: {final_external_count}")
if final_external_count == 0:
    print("✓ Conversion successful! All data is now internal.")
else:
    print("⚠ Conversion may not be complete.")

# 显示文件信息
file_size = os.path.getsize(model_path)
print(f"Final ONNX file size: {file_size / (1024*1024):.2f} MB")