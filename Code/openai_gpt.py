# from openai import OpenAI
# import os
#
# def export_api_key(port):
#     os.environ['HTTP_PROXY'] = 'http://127.0.0.1:{}'.format(port)
#     os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:{}'.format(port)
#     os.environ['FTP_PROXY'] = 'http://127.0.0.1:{}'.format(port)
#     os.environ['ALL_PROXY'] = 'http://127.0.0.1:{}'.format(port)
#     os.environ['NO_PROXY'] = '127.0.0.1,localhost'
#     os.environ['OPENAI_API_KEY'] = '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'
#
# export_api_key(9999)
# client = OpenAI()
# print("hello")
# completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     # messages=[
#     #   {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     #   {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#     # ]
#     messages = [
#         {"role": "user", "content": "Hello"}
#     ]
# )
#
# print(completion.choices[0].message.content)
# # print(completion['choices'][0]['message']['content'])
#



from utils.utils import export_api_key
import requests
import json

export_api_key(9999)
url = "https://api.kuaiwenwen.top/v1/chat/completions"

# prompt = '''
# 请你根据以下问答，写出一个笑话。这个笑话的字数在100字左右。然后，请指出这一个笑话里最好笑的是哪句话。最后，请说明为什么这一句话是最好笑的。
# question: 以后两个人在一重要的是什么？
# response: 当这婚还没结。
#
# 请你仿照下面的格式生成数据。
# [笑话]：xxx
# [最好笑的话]：xxx
# [原因]：xxx
#
# 请开始生成：
# '''

prompt = "hello"

payload = json.dumps({
   "model": "gpt-3.5-turbo",  # "gpt-3.5-turbo-16k", "gpt4",
   "messages": [
      {
         "role": "system",
         "content": "You are a helpful assistant."
      },
      {
         "role": "user",
         "content": prompt
      }
   ],
   "temperature": 1,
   "max_tokens": 1024,
   "top_p": 1,
   "frequency_penalty": 0,
   "presence_penalty": 0
})
headers = {
   'Accept': 'application/json',
   'Authorization': 'Bearer sk-1xTAHfoeUiTfTo5227Ae008c43Fe4f44Ba1aA88d9074Eb4d',
   'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

# 解析 JSON 数据为 Python 字典
data = response.json()

# print(data)
# 获取 content 字段的值
content = data['choices'][0]['message']['content']

print(content)


