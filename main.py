import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from database import (
    init_db, get_sessions_from_db, add_session_to_db, delete_session_from_db,
    add_history_to_db, get_history_from_db, add_file_to_db
)

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
from langchain_core.documents import Document

import docx
import uuid

# -------------------- ENV --------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# -------------------- EMBEDDINGS --------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------- PATHS --------------------
VECTORSTORE_ROOT = "chroma_sessions"
UPLOAD_DIR = "uploaded_files"

os.makedirs(VECTORSTORE_ROOT, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------- DB --------------------
db_conn = init_db()

# -------------------- MEMORY --------------------
store = {}
vectorstores = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in store:
        store[session] = ChatMessageHistory()
    return store[session]

def vectorstore_dir_for_session(session_id: str) -> str:
    return os.path.join(VECTORSTORE_ROOT, session_id)

def load_vectorstore(session_id: str):
    directory = vectorstore_dir_for_session(session_id)
    if os.path.isdir(directory):
        try:
            return Chroma(persist_directory=directory, embedding_function=embeddings)
        except:
            return None
    return None

def save_vectorstore(session_id: str, vectorstore_obj):
    if hasattr(vectorstore_obj, "persist"):
        vectorstore_obj.persist()
    vectorstores[session_id] = vectorstore_obj

# -------------------- DOCX LOADER --------------------
def load_docx_file(file_path):
    doc = docx.Document(file_path)
    full_text = []

    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                full_text.append(cell.text)

    content = "\n".join(full_text)
    return [Document(page_content=content, metadata={"source": file_path})]

# -------------------- FASTAPI --------------------
app = FastAPI(title="Graspd API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MODELS --------------------
class SessionCreate(BaseModel):
    session_id: str

class ChatQuery(BaseModel):
    query: str

class MessageHistory(BaseModel):
    role: str
    content: str
    timestamp: str

# -------------------- STARTUP --------------------
@app.on_event("startup")
def startup_event():
    if not get_sessions_from_db():
        add_session_to_db("default_session")

# -------------------- SESSION APIs --------------------
@app.get("/sessions", response_model=List[str])
def get_sessions():
    return get_sessions_from_db()

@app.post("/sessions")
def create_session(session: SessionCreate):
    session_id = session.session_id.strip()

    if not session_id:
        raise HTTPException(400, "Session name cannot be empty")

    if session_id in get_sessions_from_db():
        raise HTTPException(400, "Session already exists")

    add_session_to_db(session_id)
    get_session_history(session_id)

    return {"message": "Session created", "session_id": session_id}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id not in get_sessions_from_db():
        raise HTTPException(404, "Session not found")

    store.pop(session_id, None)
    vectorstores.pop(session_id, None)

    delete_session_from_db(session_id)

    vector_dir = vectorstore_dir_for_session(session_id)
    if os.path.isdir(vector_dir):
        shutil.rmtree(vector_dir)

    return {"message": f"{session_id} deleted"}

# -------------------- UPLOAD API --------------------
@app.post("/sessions/{session_id}/upload")
async def upload_documents(
    session_id: str,
    files: List[UploadFile] = File(...)
):
    if session_id not in get_sessions_from_db():
        raise HTTPException(404, "Session not found")

    documents = []
    failed_files = []

    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_upload_dir, exist_ok=True)

    for uploaded_file in files:
        file_ext = uploaded_file.filename.split('.')[-1].lower()

        # ✅ Restrict file types
        if uploaded_file.content_type not in [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]:
            failed_files.append({
                "filename": uploaded_file.filename,
                "error": "Invalid file type"
            })
            continue

        safe_filename = f"{uuid.uuid4().hex}_{uploaded_file.filename}"
        file_path = os.path.join(session_upload_dir, safe_filename)

        with open(file_path, "wb") as f:
            f.write(await uploaded_file.read())

        try:
            if file_ext == "pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()

            elif file_ext == "docx":
                docs = load_docx_file(file_path)

            else:
                raise Exception("Unsupported format")

            documents.extend(docs)
            add_file_to_db(session_id, uploaded_file.filename, file_path)

        except Exception as e:
            os.remove(file_path)
            failed_files.append({
                "filename": uploaded_file.filename,
                "error": str(e)
            })

    if not documents:
        raise HTTPException(400, {"errors": failed_files})

    # Split + Embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = splitter.split_documents(documents)

    vectorstore_dir = vectorstore_dir_for_session(session_id)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=vectorstore_dir
    )

    save_vectorstore(session_id, vectorstore)

    return {
        "message": "Upload successful",
        "chunks": len(splits),
        "failed_files": failed_files
    }

# -------------------- RAG --------------------
def get_rag_chain_for_session(session_id: str):
    if not api_key:
        raise Exception("Missing GROQ_API_KEY")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

    vs = vectorstores.get(session_id) or load_vectorstore(session_id)

    if vs is None:
        return None

    vectorstores[session_id] = vs
    retriever = vs.as_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase question with history"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using context. Max 3 sentences.\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever, qa_chain
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

# -------------------- CHAT --------------------
@app.post("/sessions/{session_id}/chat")
def chat(session_id: str, query: ChatQuery):
    if session_id not in get_sessions_from_db():
        raise HTTPException(404, "Session not found")

    chain = get_rag_chain_for_session(session_id)

    if chain is None:
        raise HTTPException(400, "Upload documents first")

    user_input = query.query.strip()
    if not user_input:
        raise HTTPException(400, "Empty query")

    add_history_to_db(session_id, "user", user_input)

    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    answer = response.get("answer", "")

    add_history_to_db(session_id, "assistant", answer)

    return {"user": user_input, "assistant": answer}

# -------------------- HISTORY --------------------
@app.get("/sessions/{session_id}/history", response_model=List[MessageHistory])
def get_history(session_id: str):
    if session_id not in get_sessions_from_db():
        raise HTTPException(404, "Session not found")

    rows = get_history_from_db(session_id)

    return [
        {
            "role": r["role"],
            "content": r["content"],
            "timestamp": r["timestamp"]
        }
        for r in rows
    ]