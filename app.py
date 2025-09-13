import os
from uuid import uuid4
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


DATA_PATH = os.path.join(os.getcwd(), "MedicalQuestionAnswering.csv")
PERSIST_DIR = "chroma_index" 

def load_docs(csv_path: str):
    if not os.path.exists(csv_path):
        st.error(f"CSV not found at:\n{csv_path}")
        st.stop()
    loader = CSVLoader(file_path=csv_path, csv_args={"delimiter": ","}, encoding="utf-8")
    return loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)

def build_or_load_vectorstore(docs):
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    splits = splitter.split_documents(docs)
    vs = Chroma.from_documents(splits, embedding=embeddings, persist_directory=PERSIST_DIR)
    vs.persist()
    return vs

# SETUP RAG CHAIN
if "retriever" not in st.session_state:
    docs = load_docs(DATA_PATH)
    vstore = build_or_load_vectorstore(docs)
    st.session_state.retriever = vstore.as_retriever()

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference "
     "context in the chat history, formulate a standalone question that can be "
     "understood without the chat history. Do NOT answer; only rewrite if needed."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_system = (
    "You are a Medical FAQ assistant. Use ONLY the retrieved context below to answer the user's question.\n\n"
    "Rules:\n"
    "1) If the answer is not in the context, say: \"I don't know based on the current knowledge base.\"\n"
    "2) Do not invent information or provide personal medical advice.\n"
    "3) Keep answers concise (2–4 sentences).\n"
    "4) Use clear, simple language. If multiple pieces of context are relevant, synthesize them.\n"
    "5) If the question is outside general medical FAQs (e.g., diagnosis, prescriptions, emergencies), "
    "respond with a safety note advising the user to consult a healthcare professional.\n\n"
    "Retrieved context:\n{context}"
)


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm, st.session_state.retriever, contextualize_q_prompt
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# session data
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "store" not in st.session_state:
    st.session_state.store = {}
    
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_session_history(s: str) -> BaseChatMessageHistory:
    if s not in st.session_state.store:
        st.session_state.store[s] = ChatMessageHistory()
    return st.session_state.store[s]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Chat UI
st.title("Medical FAQ Chatbot")


col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Clear chat"):
        st.session_state.messages = []

        sid = st.session_state.session_id
        if sid in st.session_state.store:
            st.session_state.store[sid].clear()
        st.rerun()


for role, content in st.session_state.messages[-30:]:
    prefix = "🧑‍💻 You:" if role == "user" else "🤖 Assistant:"
    st.markdown(f"**{prefix}** {content}")


user_text = st.chat_input("Ask a question…")
if user_text:

    st.session_state.messages.append(("user", user_text))

    with st.spinner("Thinking…"):
        resp = conversational_rag.invoke(
            {"input": user_text},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )
        answer = resp["answer"]


    st.session_state.messages.append(("assistant", answer))
    st.markdown(f"**🤖 Assistant:** {answer}")

