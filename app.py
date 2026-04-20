# ============================
# 网页版 RAG 智能聊天机器人
# 100% 完整代码 + 界面 + 上线
# ============================
import streamlit as st
from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from dotenv import load_dotenv
from google import genai

# 页面配置
st.set_page_config(page_title="RAG 知识库", page_icon="🤖")
st.title("🤖 RAG 智能知识库聊天机器人")

# 缓存模型（只加载一次）
@st.cache_resource
def load_models():
    load_dotenv()
    embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    google_client = genai.Client()

    # 向量库
    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(name="rag_db")
    return embedding_model, cross_encoder, google_client, coll

embedding_model, cross_encoder, google_client, coll = load_models()

# ======================
# RAG 核心函数
# ======================
def split_into_chunks(doc_file):
    with open(doc_file, "r", encoding="utf-8") as f:
        return [c for c in f.read().split("\n\n") if c.strip()]

def embed(chunk):
    return embedding_model.encode(chunk, normalize_embeddings=True).tolist()

def build_db(file):
    chunks = split_into_chunks(file)
    embeddings = [embed(c) for c in chunks]
    for i, (c, e) in enumerate(zip(chunks, embeddings)):
        coll.add(documents=[c], embeddings=[e], ids=[str(i)])
    return chunks

def retrieve(query, top=5):
    q_emb = embed(query)
    res = coll.query(query_embeddings=[q_emb], n_results=top)
    return res["documents"][0]

def rerank(query, chunks, top=3):
    pairs = [(query, c) for c in chunks]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in scored][:top]

def answer(query, chunks):
    prompt = f"""根据资料回答，不要编造。
问题：{query}
资料：{"\n\n".join(chunks)}"""
    resp = google_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return resp.text

# ======================
# 初始化知识库（只执行一次）
# ======================
if "db_ready" not in st.session_state:
    with st.spinner("正在加载知识库..."):
        build_db("代码交接.md")  # 你的文档
        st.session_state.db_ready = True

# ======================
# 聊天界面
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
user_input = st.chat_input("请输入你的问题...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG 流程
    with st.spinner("思考中..."):
        retrieved = retrieve(user_input)
        reranked = rerank(user_input, retrieved)
        ans = answer(user_input, reranked)

    # 显示回答
    with st.chat_message("assistant"):
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})