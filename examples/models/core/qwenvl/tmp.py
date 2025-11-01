# Generate the Vision Transformer (ViT) ONNX model and the TensorRT engine.
# 第一步：重新导出ONNX模型
python3 examples/models/core/qwenvl/vit_onnx_trt.py \
    --onnxFile ./model_zoo_trt_llm/qwen_vl_onnx/visual_encoder.onnx \
    --pretrained_model_path ./model_zoo_trt_llm/Qwen-VL-Chat \
    --planFile ./model_zoo_trt_llm/qwen_vl_plane/visual_encoder_fp16.plan

# 检查生成的ONNX文件
ls -la ./model_zoo_trt_llm/qwen_vl_onnx/

# 第二步：构建TensorRT引擎
python3 examples/models/core/qwenvl/vit_onnx_trt.py \
    --only_trt \
    --onnxFile ./model_zoo_trt_llm/qwen_vl_onnx/visual_encoder.onnx \
    --planFile ./model_zoo_trt_llm/qwen_vl_plane/visual_encoder_fp16.plan

# -----------------------------------------------------------


python3 examples/models/core/qwenvl/vit_onnx_trt.py --onnxFile ./model_zoo_trt_llm/qwen_vl_onnx/visual_encoder.onnx \
    --planFile ./model_zoo_trt_llm/qwen_vl_plane/visual_encoder_fp16.plan \
    --pretrained_model_path ./model_zoo_trt_llm/Qwen-VL-Chat

# Convert
python3 ./examples/models/core/qwen/convert_checkpoint.py --model_dir=./model_zoo_trt_llm/Qwen-VL-Chat \
        --output_dir=./model_zoo_trt_llm/tllm_checkpoint_1gpu \
        --dtype float16

# Build TensorRT LLM engine
trtllm-build --checkpoint_dir=./model_zoo_trt_llm/tllm_checkpoint_1gpu \
             --gemm_plugin=float16 --gpt_attention_plugin=float16 \
             --max_input_len=2048 --max_seq_len=3072 \
             --max_batch_size=8 --max_prompt_embedding_table_size=2048 \
             --remove_input_padding=enable \
             --output_dir=./model_zoo_trt_llm/trt_engines/Qwen-VL-7B-Chat
             
#  4.1 Run with INT4 GPTQ weight-only quantization engine
python3 examples/models/core/qwenvl/run.py \
    --tokenizer_dir=./Qwen-VL-Chat \
    --qwen_engine_dir=./model_zoo_trt_llm/trt_engines/Qwen-VL-7B-Chat \
    --vit_engine_path=./plan/visual_encoder/visual_encoder_fp16.plan \
    --images_path='{"image": "./pics/demo.jpeg"}'            
             