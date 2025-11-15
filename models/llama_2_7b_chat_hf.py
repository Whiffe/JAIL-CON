from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 全局变量存储模型和分词器
_model = None
_tokenizer = None
_model_name = "/home/winstonYF/attack/nanoGCG/model/shakechen/Llama-2-7b-chat-hf"  # Llama-2模型路径


def init_llama_2_7b_chat_hf():
    """初始化Llama-2-7b-chat-hf模型（仅在首次调用时执行）"""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print(f"正在加载Llama-2模型: {_model_name}...")
        _tokenizer = AutoTokenizer.from_pretrained(_model_name)
        # 设置pad_token为eos_token（Llama模型默认没有pad_token）
        _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Llama-2-7b-chat-hf模型加载完成")


def call_llama_2_7b_chat_hf(system_msg, user_msg, enable_thinking=False):
    """调用Llama-2-7b-chat-hf模型生成回应（复用全局模型）
    注：Llama-2模型不支持思维链模式，enable_thinking参数无效
    """
    try:
        # 确保模型已初始化
        init_llama_2_7b_chat_hf()
        
        # 构建符合Llama-2对话格式的输入
        prompt = f"""<s>[INST] <<SYS>>
{system_msg}
<</SYS>>

{user_msg} [/INST]"""
        
        # 准备输入
        model_inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(_model.device)
        
        # 生成文本
        generated_ids = _model.generate(
            **model_inputs,
            max_new_tokens=32768,
            temperature=0.7,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id
        )
        
        # 解析结果
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = _tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        # 忽略enable_thinking参数，始终只返回内容
        return content
        
    except Exception as e:
        print(f"Llama-2-7b-chat-hf模型调用出错: {str(e)}")
        return None