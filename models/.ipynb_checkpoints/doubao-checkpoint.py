import os
from openai import OpenAI

def call_doubao(system_msg, user_msg):
    """调用豆包API生成回应"""
    messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
    try:
        # 初始化客户端
        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.environ.get("ARK_API_KEY"),
        )

        # 调用API
        response = client.chat.completions.create(
            model="ep-20250901135540-mgv2k",  # 替换为您的方舟推理接入点ID
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
        )

        # 返回生成的内容
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"豆包API调用出错: {str(e)}")
        return None
