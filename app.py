import streamlit as st
import os
import chromadb
from dotenv import load_dotenv
import requests

# 页面配置
st.set_page_config(page_title="RAG知识库", page_icon="🤖")
st.title("🤖 RAG智能知识库聊天机器人")

# 加载API Key
load_dotenv()

# 初始化向量库（使用chromadb自带的嵌入模型，不依赖sentence-transformers）
client = chromadb.EphemeralClient()
coll = client.get_or_create_collection(name="rag_db")

# 1. 把文档分块并存入向量库
def build_db(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        if not chunks:
            return
        coll.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
    except Exception as e:
        st.error(f"加载文档失败：{str(e)}")

# 2. 向量检索
def retrieve(query, top=3):
    results = coll.query(query_texts=[query], n_results=top)
    return results["documents"][0] if results["documents"] else []

# 3. 调用豆包API
def doubao_answer(query, chunks):
    api_key = os.getenv("DOUBAO_API_KEY")
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "doubao-1-5-lite-32k-250115",
        "messages": [
            {
                "role": "system",
                "content": "你是一个知识库问答助手，只能根据用户提供的资料回答问题。如果资料里没有相关信息，就说‘未找到相关内容’，不要反问用户，也不要让用户明确问题。"
            },
            {
                "role": "user",
                "content": f"资料：{' '.join(chunks)}\n问题：{query}\n请直接根据资料回答问题，不要说无关的话。"
            }
        ]
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)
        r.raise_for_status()
        result = r.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "未找到相关内容"
    except Exception as e:
        return f"调用出错：{str(e)}"

# 初始化知识库
if "db_ready" not in st.session_state:
    with st.spinner("正在加载知识库..."):
        build_db("代码交接.md")
        st.session_state.db_ready = True

# 聊天界面
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("请输入问题...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("思考中..."):
        retrieved_chunks = retrieve(user_input)
        answer = doubao_answer(user_input, retrieved_chunks)

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})