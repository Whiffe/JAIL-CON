from modelscope import AutoModelForCausalLM, AutoTokenizer
import re
import torch
import os

# 全局缓存：key为模型规格（0.6B/4B/8B），value为Qwen3Guard实例
_guard_instances = {}

class Qwen3Guard:
    def __init__(self, model_size):
        """初始化Qwen3Guard模型（仅首次创建时加载）"""
        bash_path = "/home/winstonYF/attack/nanoGCG/model/Qwen/"
        model_map = {
            "0.6B": os.path.join(bash_path, "Qwen3Guard-Gen-0.6B"),
            "4B": os.path.join(bash_path, "Qwen3Guard-Gen-4B"),
            "8B": os.path.join(bash_path, "Qwen3Guard-Gen-8B")
        }
        self.model_name = model_map.get(model_size, "Qwen/Qwen3Guard-Gen-0.6B")
        print(f"正在加载Qwen3Guard模型 {model_size}: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print(f"Qwen3Guard模型 {model_size} 加载完成")

    def extract_label_and_categories(self, content):
        """提取安全标签和类别"""
        safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
        category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
        
        safe_label_match = re.search(safe_pattern, content)
        label = safe_label_match.group(1) if safe_label_match else None
        
        categories = re.findall(category_pattern, content)
        return label, categories

    def judge(self, prompt):
        """审核输入内容并返回结果"""
        try:
            # 构建输入消息
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
            
            # 模型推理
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=128
            )
            
            # 解析输出
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # 提取标签和类别
            safe_label, categories = self.extract_label_and_categories(content)
            return {
                "safety_label": safe_label,
                "categories": categories,
                "raw_output": content
            }
            
        except Exception as e:
            print(f"Qwen3Guard 调用出错: {str(e)}")
            return None

# 便捷调用函数（复用缓存的模型实例）
def call_qwen3guard(model_size, prompt):
    """调用Qwen3Guard审核模型（优先使用缓存实例）"""
    global _guard_instances
    # 如果模型实例未缓存，则创建并缓存
    if model_size not in _guard_instances:
        _guard_instances[model_size] = Qwen3Guard(model_size)
    # 使用缓存的实例进行推理
    return _guard_instances[model_size].judge(prompt)