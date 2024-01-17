from flask import Flask
from flask import render_template
from flask import request
import os
import json
from api import ai_client, db_client, post_chat, create_embeddings

app = Flask(__name__)

# 读取json文件
with open('../config.json') as f:
    global_config = json.load(f)

def create_prompt(question, knowledge):
    bg = ""
    for i, v in enumerate(knowledge):
        bg += f"{i}. {v}"

    prompt = f"""下面为你提供了一些有助于回答问题的背景，结合背景和你所知道的知识回答下列问题：
        背景：{bg}
        问题：{question}"""
    return prompt


def query(text):
    """
    执行逻辑：
    首先使用openai的Embedding API将输入的文本转换为向量
    然后使用Qdrant的search API进行搜索，搜索结果中包含了向量和payload
    payload中包含了title和text，title是疾病的标题，text是摘要
    最后使用openai的ChatCompletion API进行对话生成
    """
    collection_name = global_config['COLLECTION_NAME']
    sentence_embedding = create_embeddings(text)
    search_result = db_client.search(
        collection_name=collection_name,
        query_vector=sentence_embedding,
        limit= 5,
        search_params={"exact": False, "hnsw_ef": 128}
    )
    # print("查找结果：", search_result)

    answers = [res.payload["text"] for res in search_result]

    completion = post_chat(create_prompt(text, answers))

    return completion


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']

    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": query(search),
        },
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)