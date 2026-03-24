import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load env
load_dotenv()

# Set HF token
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# UI
st.title("Graspd")

# Load API key
api_key = os.getenv("GROQ_API_KEY")

# ---------------- SESSION FUNCTION ---------------- #
def get_session_history(session: str) -> BaseChatMessageHistory:
    if 'store' not in st.session_state:
        st.session_state.store = {}

    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()

    return st.session_state.store[session]


# ---------------- MAIN APP ---------------- #
if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

    session_id = st.text_input("Session ID", value="default_session")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True
    )

    conversational_rag_chain = None  # safety

    # ---------------- PROCESS PDFs ---------------- #
    if uploaded_files:
        documents = []

        for uploaded_file in uploaded_files:
            temp_path = f"./temp_{uploaded_file.name}"

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)

        # Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )
        splits = text_splitter.split_documents(documents)

        # Vector DB
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

        # ---------------- CONTEXTUAL RETRIEVER ---------------- #
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given chat history and latest question, reformulate into standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # ---------------- QA PROMPT ---------------- #
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an assistant. Use context to answer. If unknown, say so. Max 3 sentences.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # ---------------- FINAL CHAIN ---------------- #
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    # ---------------- CHAT ---------------- #
    user_input = st.text_input("Ask something from the PDF:")

    if user_input:
        if conversational_rag_chain is None:
            st.warning("⚠️ Please upload PDF first")
        else:
            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.write("### 🧑 You:")
            st.write(user_input)

            st.write("### 🤖 Assistant:")
            st.write(response["answer"])

else:
    st.warning("Please set GROQ_API_KEY in .env")