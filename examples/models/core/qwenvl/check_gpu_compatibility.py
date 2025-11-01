# check_gpu_compatibility.py
import tensorrt as trt
import torch

def check_compatibility():
    # 检查CUDA和TensorRT
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
    
    # 检查TensorRT版本
    print("TensorRT version:", trt.__version__)
    
    # 检查TensorRT功能
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    print("FP16 support:", builder.platform_has_fast_fp16)
    print("INT8 support:", builder.platform_has_fast_int8)
    print("FP8 support:", hasattr(builder, 'platform_has_fast_fp8') and builder.platform_has_fast_fp8)

if __name__ == "__main__":
    check_compatibility()