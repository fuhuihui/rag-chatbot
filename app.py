# ============================
# 国内 100% 可用 RAG 聊天机器人（豆包API）
# 无报错、不繁忙、秒回
# ============================
import streamlit as st
import os
from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from dotenv import load_dotenv
import requests

# 加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

st.set_page_config(page_title="RAG知识库", page_icon="🤖")
st.title("🤖 RAG智能知识库聊天机器人")

@st.cache_resource
def load_models():
    load_dotenv()
    embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    return embedding_model, cross_encoder

embedding_model, cross_encoder = load_models()

client = chromadb.EphemeralClient()
coll = client.get_or_create_collection(name="rag_db")

# ======================
# RAG 核心函数
# ======================
def split_into_chunks(doc_file):
    try:
        with open(doc_file, "r", encoding="utf-8") as f:
            return [c for c in f.read().split("\n\n") if c.strip()]
    except:
        return []

def embed(chunk):
    return embedding_model.encode(chunk, normalize_embeddings=True).tolist()

def build_db(file):
    chunks = split_into_chunks(file)
    if not chunks: return
    embeddings = [embed(c) for c in chunks]
    for i, (c, e) in enumerate(zip(chunks, embeddings)):
        coll.add(documents=[c], embeddings=[e], ids=[str(i)])

def retrieve(query, top=5):
    q_emb = embed(query)
    res = coll.query(query_embeddings=[q_emb], n_results=top)
    return res["documents"][0] if res["documents"] else []

def rerank(query, chunks, top=3):
    if not chunks: return []
    pairs = [(query, c) for c in chunks]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in scored][:top]

# ======================
# ✅ 豆包 API（国内秒回）
# ======================
def doubao_answer(query, chunks):
    api_key = os.getenv("DOUBAO_API_KEY")
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 把提示词改成下面这个版本，让模型直接回答
    data = {
        "model": "doubao-1-5-lite-32k-250115",
        "messages": [
            {
                "role": "system",
                "content": "你是一个知识库问答助手，只能根据用户提供的资料来回答问题。如果资料里没有相关信息，就说‘未找到相关内容’，不要反问用户，也不要让用户明确问题。"
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

# ======================
# 初始化
# ======================
if "db_ready" not in st.session_state:
    with st.spinner("正在加载知识库..."):
        build_db("代码交接.md")
        st.session_state.db_ready = True

# ======================
# 聊天界面
# ======================
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
        retrieved = retrieve(user_input)
        reranked = rerank(user_input, retrieved)
        ans = doubao_answer(user_input, reranked)

    with st.chat_message("assistant"):
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})