import os
import openai
import json

def call_gemini_2_5(system_msg, user_msg, enable_thinking=False):
    """调用gemini-2.5-flash模型生成回应"""
    try:
        # 初始化客户端
        client = openai.OpenAI(
            api_key="sk-xxxx",  # 替换为实际密钥
            base_url="https://aihubmix.com/v1"
        )

        # 构建消息列表
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        # 调用API（移除思维链相关参数）
        completion = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages
        )

        # 解析并返回结果
        result = json.loads(completion.model_dump_json())
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"gemini-2.5-flash调用出错: {str(e)}")
        return None