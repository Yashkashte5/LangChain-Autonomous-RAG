# app.py

import streamlit as st
import os
from src.rag_chain import RAG

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")

# --- CREATE FOLDERS IF NOT EXIST ---
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/vectorstore", exist_ok=True)

# --- INIT SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag" not in st.session_state:
    st.session_state.rag = RAG()

rag = st.session_state.rag

st.title("💬 RAG Chat ")
st.write("Upload documents and ask questions based on them!")

# --- FILE UPLOAD ---
uploaded_files = st.file_uploader(
    "Upload one or more files", 
    type=["pdf", "txt", "docx", "csv", "md"], 
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        save_path = os.path.join("data/raw", file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
    
    # Build vectorstore only once per session
    if not rag.db:
        rag.build_vectorstore(folder_path="data/raw")

# --- CHAT INPUT WITH CALLBACK ---
def submit_callback():
    user_input = st.session_state.input_text.strip()
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        response = rag.query(user_input)
        st.session_state.chat_history.append(("ai", response))
        st.session_state.input_text = ""  # safely reset input

st.text_input(
    "Ask something...",
    key="input_text",
    placeholder="Type your message here...",
    label_visibility="collapsed",
    on_change=submit_callback
)

# --- DISPLAY CHAT HISTORY WITH WHATSAPP-STYLE UI ---
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f"""
            <div style="
                background-color:#DCF8C6;
                color:#000;
                padding:10px;
                border-radius:15px;
                width:fit-content;
                max-width:70%;
                margin-left:auto;
                margin-bottom:5px;
                font-size:14px;
                word-wrap:break-word;
            ">
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                background-color:#FFFFFF;
                color:#000;
                padding:10px;
                border-radius:15px;
                width:fit-content;
                max-width:70%;
                margin-right:auto;
                margin-bottom:5px;
                border:1px solid #E0E0E0;
                font-size:14px;
                word-wrap:break-word;
            ">
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )

# --- STYLE FIXES FOR INPUT BOX ---
st.markdown(
    """
    <style>
    div.stTextInput>div>div>input {
        height: 40px;
        font-size: 14px;
        padding-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


import warnings
from langchain import LangChainDeprecationWarning 

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
