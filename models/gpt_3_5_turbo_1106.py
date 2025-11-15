import os
import openai
import json

def call_gpt_3_5_turbo_1106(system_msg, user_msg, enable_thinking=False):
    """调用GPT-3.5-turbo-1106模型生成回应（不支持思维链模式）"""
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
            # model="gpt-3.5-turbo-1106",
            model="gpt-3.5-turbo",
            messages=messages
        )

        # 解析并返回结果
        result = json.loads(completion.model_dump_json())
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"GPT-3.5-turbo-1106调用出错: {str(e)}")
        return None