import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langdetect import detect
import numpy as np
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

st.title("RAG Application using Sentence Transformers (Free)")

# File uploader for user PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp_uploaded.pdf")
    data = loader.load()
else:
    st.info("Please upload a PDF file to start.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Use HuggingFace sentence-transformers for free embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load QA pipeline (do this once, outside the query block)
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

qa_pipeline = load_qa_pipeline()

query = st.chat_input("Ask me anything: ")

if query:
    user_lang = detect(query)
    # Retrieve relevant docs
    results = retriever.get_relevant_documents(query)
    if results:
        # Concatenate top 3 chunks for context
        context = " ".join([doc.page_content for doc in results])
        # Use QA model to answer
        qa_input = {"question": query, "context": context}
        answer = qa_pipeline(qa_input)["answer"]
    else:
        answer = "Sorry, I couldn't find an answer in the document."

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})