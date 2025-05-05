import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import json
import tempfile
import os

# 1. Tải model embedding
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

st.set_page_config(page_title="Hệ thống Hỏi đáp AI", layout="wide")
st.title("🦉 Hệ thống Hỏi đáp Thông minh cho Tài liệu PDF hoặc Chatbot")

# 2. Hàm xử lý file PDF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

# 3. Hàm tạo FAISS index
def create_faiss_index(embeddings):
    quantizer = faiss.IndexFlatL2(embeddings.shape[1])
    nlist = min(50, len(embeddings)) if len(embeddings) > 0 else 1
    if len(embeddings) < nlist:
        nlist = len(embeddings)
    if nlist < 1:
        nlist = 1
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
    index.train(embeddings)
    index.add(embeddings)
    return index

# 4. Hàm gọi LLM (nếu có context thì dùng, không thì hỏi tự do)
def query_llm(context, query):
    if context:
        system_prompt = "Bạn là chuyên gia phân tích tài liệu. Chỉ trả lời dựa trên ngữ cảnh được cung cấp."
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    else:
        system_prompt = "Bạn là trợ lý AI thông minh, hãy trả lời các câu hỏi một cách tự nhiên và chính xác."
        user_prompt = query

    payload = {
        'temperature': 0.1,
        'max_tokens': 512,
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối: {str(e)}")
        return None

# 5. Khởi tạo model embedding
model = load_embedding_model()

# 6. Cho phép upload file (hoặc không)
uploaded_file = st.file_uploader("📤 Tải lên file PDF (tùy chọn)", type="pdf", accept_multiple_files=False)

# 7. Nhập câu hỏi (luôn hiển thị, không phụ thuộc việc upload file)
query = st.text_input("💬 Nhập câu hỏi của bạn:", placeholder="Bạn có thể hỏi về tài liệu hoặc hỏi bất kỳ điều gì...")

# 8. Nếu có file, xử lý file và tạo index
faiss_index = None
texts = None
if uploaded_file:
    with st.spinner("🔍 Đang xử lý tài liệu..."):
        pdf_path = process_pdf(uploaded_file)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        os.unlink(pdf_path)  # Xóa file tạm sau khi đọc xong
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(docs)
        if len(texts) > 0:
            text_embeddings = model.encode([text.page_content for text in texts], show_progress_bar=False)
            faiss_index = create_faiss_index(text_embeddings)

# 9. Khi người dùng nhập câu hỏi
if query:
    with st.spinner("🤖 Đang phân tích..."):
        context = ""
        # Nếu đã upload file và đã tạo index, thực hiện RAG
        if uploaded_file and faiss_index is not None and texts is not None and len(texts) > 0:
            query_embedding = model.encode(query, show_progress_bar=False)
            D, I = faiss_index.search(np.array([query_embedding]), k=min(3, len(texts)))
            context = "\n---\n".join([texts[i].page_content for i in I[0]])
        # Nếu không có file, context để trống, AI sẽ trả lời tự do
        answer = query_llm(context, query)
        if answer:
            st.subheader("📝 Câu trả lời:")
            st.success(answer)
            if context:
                with st.expander("Xem ngữ cảnh tham khảo"):
                    st.markdown(context)
        else:
            st.warning("Không tìm thấy thông tin phù hợp.")
