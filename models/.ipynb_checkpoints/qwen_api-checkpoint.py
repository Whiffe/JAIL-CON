import os
from openai import OpenAI
import json

def call_qwen_api(system_msg, user_msg):
    """调用通义千问API生成回应"""
    try:
        # 初始化客户端
        client = OpenAI(
            # api_key=os.getenv("DASHSCOPE_API_KEY"),
            api_key="sk-2d0ca666354540b68ec195ff84c386c2",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 调用API
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            extra_body={"enable_thinking": False},
        )
        
        # 解析并返回生成的内容
        result = json.loads(completion.model_dump_json())
        return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        print(f"通义千问API调用出错: {str(e)}")
        return None
