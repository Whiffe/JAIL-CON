from modelscope import AutoModelForCausalLM, AutoTokenizer


# 全局变量存储模型和分词器
_model = None
_tokenizer = None
_model_name = "/home/winstonYF/attack/nanoGCG/model/Qwen/Qwen3-1.7B"  # 1.7B模型路径


def init_qwen_open_1_7B():
    """初始化Qwen开源1.7B模型（仅在首次调用时执行）"""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print(f"正在加载Qwen开源模型: {_model_name}...")
        _tokenizer = AutoTokenizer.from_pretrained(_model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Qwen开源1.7B模型加载完成")


def call_qwen_open_1_7B(system_msg, user_msg, enable_thinking=False):
    """调用开源Qwen 1.7B模型生成回应（复用全局模型）添加enable_thinking参数，默认值为False"""
    try:
        # 确保模型已初始化
        init_qwen_open_1_7B()
        
        # 准备输入
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        model_inputs = _tokenizer([text], return_tensors="pt").to(_model.device)
        
        # 生成文本
        generated_ids = _model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # 解析结果
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
            

        thinking_content = _tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = _tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        if enable_thinking:
            return thinking_content, content  # 返回两个值
        else:
            return content
        
    except Exception as e:
        print(f"开源Qwen 4B模型调用出错: {str(e)}")
        return None, None