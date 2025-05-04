import os  
import base64
from openai import AzureOpenAI  

endpoint = os.getenv("ENDPOINT_URL")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

class Llm:
    def __init__(self):
        self.client = AzureOpenAI(  
            azure_endpoint = endpoint,  
            api_key = subscription_key,  
            api_version = "2024-05-01-preview",
        )
        
    def generate(self, query, system_prompt = "你是一个有用的人工智能助手。"):
        completion = self.client.chat.completions.create(
            model = deployment,
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content":[
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
            ],
            logprobs = True
        )
        response = completion.choices[0].message.content
        logprobs = completion.choices[0].logprobs # 无法返回完整输出分布
        return response, logprobs
    
if __name__ == '__main__':
    gpt4o = Llm()
    query = "请介绍一下北理工。"
    response, logprobs = gpt4o.generate(query)
    print(response)
    print(logprobs)
