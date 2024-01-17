from qdrant_client import QdrantClient
import openai
from word2vec import Word2VecManager
import requests
from os import path
import json

# 读取json文件
with open('../config.json') as f:
    global_config = json.load(f)

def load_stopwords(file_path):
  # 读取停词表，并使用set来存储
  with open(file_path, 'r', encoding='utf-8') as file:
      stopwords_set = set(line.strip() for line in file)
      return stopwords_set

def create_ai_client():
  client = openai.AzureOpenAI(
      azure_endpoint="https://search.bytedance.net/gpt/openapi/online/v2/crawl",
      api_version="2023-07-01-preview",
      api_key=global_config['OPENAI_API_KEY']
  )
  return client

def create_db_client():
  db_client = QdrantClient("127.0.0.1", port=6333)
  return db_client

stop_word_set = load_stopwords(path.join('../', global_config['STOPWORDS_PATH']))           
ai_client = create_ai_client()
db_client = create_db_client()
word2vec_model = Word2VecManager(model_name_or_path=path.join('../', global_config['WORD2VEC_PATH']),stopwords=stop_word_set)

def requestChatGLM2(data):
  headers = {
    'Content-Type': 'application/json'
  }
  port = global_config['CHATGLM_PORT']
  response = requests.post(f'http://127.0.0.1:{port}', headers=headers, data=json.dumps(data))
  return response.json()

# 根据模型类型来进行embedding
# 如果是openai，就调用openai的text-embedding-ada-002
# 如果是word2vec，就使用word2vec.py里的模型
def create_embeddings(text):
  model_type = global_config['EMBEDDING_MODEL_TYPE']
  if model_type == "word2vec":
    return word2vec_model.encode(text)
  elif model_type == "openai":
    sentence_embeddings = ai_client.embeddings.create(
      model="text-embedding-ada-002",
      input=text,
    )
    # print("成功向量化，token数：{}".format(sentence_embeddings.usage.total_tokens))
    return sentence_embeddings.data[0].embedding
  else:
    raise Exception("未知的模型类型")

# 根据模型类型来调用对话
# 如果是openai，就调用openai的gpt系列
# 如果是chatglm，就使用自行部署的chatglm
def post_chat(prompt):
  model_type = global_config['CHAT_MODEL_TYPE']
  if model_type == "chatglm":
    data = {
      'prompt': prompt,
      'history': []
    }
    completion = requestChatGLM2(data)
    return completion["response"]
  elif model_type == "openai":
    completion = ai_client.chat.completions.create(
        model="gpt-35-turbo-16k",
        temperature=0.7,
        messages=[
          {'role': 'user', 'content': prompt},
        ]
    )
    return completion.choices[0].message.content
  else:
    raise Exception("未知的模型类型")
