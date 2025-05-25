import fitz  # PyMuPDF
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from google import genai
import tempfile
import os

# --- Set Streamlit Page Config ---
st.set_page_config(page_title="Lecture QA Assistant", layout="wide")

# --- Sidebar: Gemini API Key ---
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password", value="AIzaSyAQm7AwskmDw_wYkSVdOxVcXMj3DsWWIAw")

# --- Load Models ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embedder, cross_encoder

embedder, cross_encoder = load_models()

# --- PDF Chunking with Metadata (extract all text) ---
def extract_lecture_chunks(pdf_path, max_words=250, overlap=50):
    doc = fitz.open(pdf_path)
    chunks = []
    metadata = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            continue

        words = text.split()
        for k in range(0, len(words), max_words - overlap):
            chunk_text = " ".join(words[k:k + max_words])
            chunks.append(chunk_text)
            metadata.append({
                "lecture": "N/A",
                "page": i + 1
            })

    return chunks, metadata

# --- Title and Upload ---
st.title("📚 Lecture QA Assistant")
uploaded_file = st.file_uploader("Upload Lecture PDF", type=["pdf"])

# --- Initialize Gemini if API provided ---
if api_key and "gemini_client" not in st.session_state:
    st.session_state.gemini_client = genai.Client(api_key=api_key)

if api_key and "chat_session" not in st.session_state:
    st.session_state.chat_session = st.session_state.gemini_client.chats.create(model="gemini-2.0-flash")

# --- Process PDF only once ---
if uploaded_file and api_key:
    if "uploaded_filename" not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
        with st.spinner("Reading and processing the PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            chunks, metadata = extract_lecture_chunks(tmp_path)

            if not chunks:
                st.error("No text chunks found. Ensure the PDF has readable text.")
                os.remove(tmp_path)
                st.stop()

            embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)

            # Save to session state
            st.session_state.chunks = chunks
            st.session_state.metadata = metadata
            st.session_state.embeddings = embeddings
            st.session_state.index = index
            st.session_state.uploaded_filename = uploaded_file.name

            st.success(f"Processed {len(chunks)} chunks from {uploaded_file.name}")
            os.remove(tmp_path)

# --- Handle Queries if Data is Ready ---
if "index" in st.session_state and api_key:
    query = st.text_input("Ask a question about the lectures:")

    if query:
        chunks = st.session_state.chunks
        metadata = st.session_state.metadata
        index = st.session_state.index

        query_embedding = embedder.encode([query])
        _, indices = index.search(query_embedding, 10)

        candidates = [chunks[i] for i in indices[0]]
        meta = [metadata[i] for i in indices[0]]

        pairs = [[query, c] for c in candidates]
        scores = cross_encoder.predict(pairs)
        sorted_results = sorted(zip(scores, candidates, meta), key=lambda x: x[0], reverse=True)[:5]

        context = "\n\n".join([r[1] for r in sorted_results])
        references = "\n".join([f"Page {r[2]['page']}" for r in sorted_results])

        prompt = f"""
You are a helpful teaching assistant. Use the context below to answer the question clearly and concisely.
Always mention references from the page if relevant.

Context:
{context}

Question:
{query}
"""
        response = st.session_state.chat_session.send_message(prompt)

        st.subheader("Answer")
        st.write(response.text.strip())

        with st.expander("📖 References"):
            st.code(references)

        with st.expander("💬 Conversation History"):
            for msg in st.session_state.chat_session.get_history():
                role = "🧑‍🎓 You" if msg.role == "user" else "🤖 Gemini"
                st.markdown(f"**{role}:** {msg.parts[0].text}")

elif uploaded_file and not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to continue.")

# --- Optional Reset Button ---
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.pop("chat_session", None)
    st.success("Chat session reset.")
