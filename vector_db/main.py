from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import os
import tqdm
import json
import openai
from os import path
from word2vec import Word2VecManager


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

with open('../config.json') as f:
    global_config = json.load(f)

stop_word_set = load_stopwords(path.join('../', global_config['STOPWORDS_PATH']))              
ai_client = create_ai_client()
db_client = create_db_client()
word2vec_model = Word2VecManager(model_name_or_path=path.join('../', global_config['WORD2VEC_PATH']),stopwords=stop_word_set)

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


def split_sentences(text):
    sent_delimiters = ['。', '？', '！', '?', '!', '.', '\"', '“', '”', '\'', '‘', '’']
    for delimiter in sent_delimiters:
        text = text.replace(delimiter, '\n')
    sentences = text.split('\n')
    sentences = [sent for sent in sentences if sent.strip()]
    return sentences

if __name__ == '__main__':
    collection_name = global_config['COLLECTION_NAME']

    vc_size = 1536 if global_config['EMBEDDING_MODEL_TYPE'] == 'openai' else 300

    # 创建collection
    db_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vc_size, distance=Distance.COSINE),
    )

    count = 0
    for root, dirs, files in os.walk("../corpus"):
        for file in tqdm.tqdm(files):
          file_path = os.path.join(root, file)
          print("加载知识文档：{}\n".format(file))
          with open(file_path, 'r', encoding='utf-8') as f:
              text = f.read()
              parts = split_sentences(text)
              for part in parts:
                  print(part)
                  embedding = create_embeddings(part)
                  db_client.upsert(
                      collection_name=collection_name,
                      wait=True,
                      points=[
                          PointStruct(id=count, vector=embedding, payload={"text": part}),
                      ],
                  )
                  count += 1