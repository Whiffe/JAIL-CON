import os
from openai import OpenAI
import json

def call_deepseek(system_msg, user_msg, enable_thinking=False):
    """调用DeepSeek API生成回应"""
    try:
        # 初始化客户端
        client = OpenAI(
            # api_key=os.environ.get("DEEPSEEK_API_KEY"),
            api_key='sk-xxxx',
            base_url="https://api.deepseek.com"
        )

        # 调用API
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            stream=False
        )
        
        # 解析并返回生成的内容
        result = json.loads(completion.model_dump_json())
        return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        print(f"DeepSeek API调用出错: {str(e)}")
        return None