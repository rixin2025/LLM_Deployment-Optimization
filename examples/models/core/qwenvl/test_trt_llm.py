from tensorrt_llm import LLM, SamplingParams


def main():

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)

    print("准备加载模型...")
    # llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    llm = LLM(
        model="/home/ljc/llm_deploy_proj/trt_llm_framework/TensorRT-LLM-main/model_zoo_trt_llm/Qwen3-0.6B",
        tensor_parallel_size=1  # 使用单GPU
    )
    print("模型加载完成")

    print("开始生成文本...")
    outputs = llm.generate(prompts, sampling_params)
    print("文本生成完成")

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    print("程序开始执行")
    main()
    print("程序执行结束")