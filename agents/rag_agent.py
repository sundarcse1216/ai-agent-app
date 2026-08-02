import os

from cachetools import TTLCache
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.base_agent import BaseAgent
from config import DOCUMENT_PATH, FAISS_INDEX, MEMORY_WINDOW, OLLAMA_MODEL
from logger import logger

rag_cache = TTLCache(maxsize=100, ttl=300)


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_document(file):
    ext = os.path.splitext(file)[1].lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(file)
    elif ext == ".txt":
        loader = TextLoader(file, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} page(s) from {os.path.basename(file)}")
    return documents


def split_chunk(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


def generate_embedding():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def create_vector_db(index_name, chunks, embedding_model):
    if os.path.exists(index_name):
        logger.info("Loading existing FAISS index...")
        vector_store = FAISS.load_local(index_name, embedding_model, allow_dangerous_deserialization=True)
    else:
        logger.info("Creating new FAISS index...")
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model, normalize_L2=True)
        vector_store.save_local(index_name)
    logger.info("Vector database created successfully.")
    return vector_store


def build_rag_chain(llm, vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might reference context in the chat history, "
         "formulate a standalone question which can be understood without the chat history. "
         "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks. "
         "Use the following pieces of retrieved context to answer the question. "
         "If you don't know the answer, say that you don't know. Keep the answer concise.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


class RAGAgent(BaseAgent):
    def __init__(self):
        embedding_model = generate_embedding()

        if os.path.exists(FAISS_INDEX):
            logger.info("Loading existing FAISS index...")
            vector_store = FAISS.load_local(
                FAISS_INDEX, embedding_model, allow_dangerous_deserialization=True
            )
        else:
            doc_files = [
                f for f in os.listdir(DOCUMENT_PATH)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            ]

            if not doc_files:
                raise FileNotFoundError("No supported documents found (PDF or TXT).")

            logger.info(f"Found {len(doc_files)} document(s)")
            documents = []
            for doc in doc_files:
                documents.extend(load_document(os.path.join(DOCUMENT_PATH, doc)))

            chunks = split_chunk(documents)
            vector_store = create_vector_db(FAISS_INDEX, chunks, embedding_model)

        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        self.rag_chain = build_rag_chain(llm, vector_store)
        self.chat_history = []

    def handle(self, query):
        try:
            cache_key = query.strip().lower()
            if cache_key in rag_cache:
                logger.info("RAG cache hit")
                return rag_cache[cache_key]

            response = self.rag_chain.invoke({
                "input": query,
                "chat_history": self.chat_history
            })

            answer = response["answer"]
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=answer))
            if len(self.chat_history) > MEMORY_WINDOW * 2:
                self.chat_history = self.chat_history[-MEMORY_WINDOW * 2:]

            rag_cache[cache_key] = answer
            return answer

        except Exception:
            logger.exception("RAGAgent failed")
            return (
                "❌ Unable to retrieve information from the document.\n\n"
                "Please try again later."
            )
