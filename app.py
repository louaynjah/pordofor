import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langdetect import detect  # Add this import
import numpy as np

from dotenv import load_dotenv
load_dotenv()

st.title("RAG Application using Gemini Pro")

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

# Batching for embeddings
batch_size = 10
all_embeddings = []
batched_docs = []
# Add these imports at the top
from tenacity import retry, stop_after_attempt, wait_exponential
import time

# Add retry decorator and modify the embedding section
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_embeddings(texts, embedder):
    return embedder.embed_documents(texts)

# Modify the embedding section
embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    request_timeout=120,  # Increase timeout to 120 seconds
)

for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    texts = [doc.page_content for doc in batch]
    try:
        embeddings = get_embeddings(texts, embedder)
        all_embeddings.extend(embeddings)
        batched_docs.extend(batch)
        time.sleep(1)  # Add a small delay between batches
    except Exception as e:
        st.warning(f"Batch {i//batch_size+1} failed to embed after retries: {e}")

if not all_embeddings:
    st.error("Failed to embed any document chunks. Try a smaller PDF.")
    st.stop()

# Convert embeddings to numpy array for FAISS
all_embeddings = np.array(all_embeddings).astype("float32")

# Build FAISS vectorstore from embeddings and docs
vectorstore = FAISS.from_embeddings(
    all_embeddings,
    batched_docs
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

query = st.chat_input("Ask me anything: ")
prompt = query

if query:
    # Detect language of the user's question
    user_lang = detect(query)
    
    # Define system prompts for supported languages
    system_prompts = {
        "en": (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use six sentences maximum and keep the "
            "answer concise.\n\n{context}"
        ),
        "fr": (
            "Vous êtes un assistant pour des tâches de questions-réponses. "
            "Utilisez les éléments de contexte récupérés suivants pour répondre "
            "à la question. Si vous ne connaissez pas la réponse, dites-le. "
            "Utilisez six phrases maximum et restez concis.\n\n{context}"
        ),
        "ar": (
            "أنت مساعد لمهام الإجابة على الأسئلة. استخدم السياق التالي للإجابة "
            "على السؤال. إذا لم تعرف الإجابة، قل أنك لا تعرف. استخدم ست جمل كحد أقصى وكن موجزًا.\n\n{context}"
        ),
        # Add more languages as needed
    }
    # Choose the system prompt based on detected language, default to English
    system_prompt = system_prompts.get(user_lang, system_prompts["en"])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get model response
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})