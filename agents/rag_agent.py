import os

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.base_agent import BaseAgent
from config import (
    DOCUMENT_PATH,
    FAISS_INDEX,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)
from core.cache import make_cache
from core.cost_tracking import calculate_openai_cost
from core.semantic_router import EMBEDDING_MODEL, load_embedding_model
from logger import logger

rag_cache = make_cache()


def _turns_to_messages(turns: list[dict]) -> list:
    messages = []
    for t in turns:
        messages.append(HumanMessage(content=t["query"]))
        messages.append(AIMessage(content=t["response"]))
    return messages


class _UsageCallback(BaseCallbackHandler):
    """Captures token usage from the retrieval chain's LLM calls into an
    explicitly-held Usage (not core.cost_tracking's ambient contextvar —
    this callback fires from inside LangChain's own execution machinery,
    and Router already established that ambient context isn't reliable
    across anything but a single, non-generator call). Reads
    message.usage_metadata rather than llm_output['token_usage'] —
    confirmed by testing that the latter is OpenAI-only (None for
    ChatOllama responses), while usage_metadata is populated the same way
    for both. The chain fires on_llm_end twice per turn when there's prior
    chat history (once for query reformulation, once for the final
    answer) — both calls get summed here, which is the actually-correct
    total cost for the turn, not an undercount."""

    def __init__(self, usage, model: str, is_openai: bool):
        self.usage = usage
        self.model = model
        self.is_openai = is_openai

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                message = getattr(gen, "message", None)
                usage_metadata = getattr(message, "usage_metadata", None) if message else None
                if not usage_metadata:
                    continue
                prompt_tokens = usage_metadata.get("input_tokens", 0)
                completion_tokens = usage_metadata.get("output_tokens", 0)
                cost = (
                    calculate_openai_cost(self.model, prompt_tokens, completion_tokens)
                    if self.is_openai else 0.0
                )
                self.usage.add(prompt_tokens, completion_tokens, cost)


def _usage_config(usage) -> dict:
    if usage is None:
        return {}
    return {"callbacks": [_UsageCallback(usage, OPENAI_CHAT_MODEL, LLM_PROVIDER == "openai")]}


def build_chat_model():
    """Selects the LangChain chat model backing the RAG chain, based on
    LLM_PROVIDER. The retrieval chain (create_history_aware_retriever /
    create_stuff_documents_chain) needs a LangChain chat-model object, so
    provider selection happens at this level rather than through the
    LLMProvider abstraction used by the other agents."""
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(model=OPENAI_CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)


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


class _SharedSentenceTransformerEmbeddings(Embeddings):
    """LangChain Embeddings adapter over core.semantic_router's cached
    SentenceTransformer, instead of langchain_huggingface.HuggingFaceEmbeddings
    constructing its own separate copy of the same bge-small-en-v1.5 model.
    Two independent copies of a transformer model in one process is the kind
    of thing that fits fine on a laptop but OOMs a 512MB-RAM deploy target
    (e.g. Render's free tier) — confirmed live, not theoretical. Mirrors
    HuggingFaceEmbeddings._embed's behavior (newline stripping, no
    encode_kwargs) so the persisted FAISS index (built under the old
    embeddings class) stays compatible."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model = load_embedding_model(model_name)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        return self._model.encode(texts).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


def generate_embedding():
    return _SharedSentenceTransformerEmbeddings()


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

        llm = build_chat_model()
        self.rag_chain = build_rag_chain(llm, vector_store)

    def handle(self, query, turns=None, usage=None):
        # No conversation memory lives on `self` — the retrieval chain and
        # FAISS index above are the expensive, pooled, stateless-across-
        # sessions part (built once); turns (the whole session's shared
        # conversation history — see core/conversation_context.py) is
        # passed in by the caller, converted to LangChain messages here.
        chat_history = _turns_to_messages(turns or [])
        try:
            # NOTE: cached purely by query text, so two sessions asking the
            # same question with different prior context will share a
            # cached answer regardless of that context. Acceptable for now;
            # revisit if/when this visibly produces a wrong-context answer.
            # Also note: a cache hit means no LLM call happens, so no usage
            # is recorded for it — correct, since it genuinely cost nothing.
            cache_key = query.strip().lower()
            if cache_key in rag_cache:
                logger.info("RAG cache hit")
                return rag_cache[cache_key]

            response = self.rag_chain.invoke(
                {"input": query, "chat_history": chat_history},
                config=_usage_config(usage),
            )

            answer = response["answer"]
            rag_cache[cache_key] = answer
            return answer

        except Exception:
            logger.exception("RAGAgent failed")
            return (
                "❌ Unable to retrieve information from the document.\n\n"
                "Please try again later."
            )

    def handle_stream(self, query, turns=None, usage=None):
        chat_history = _turns_to_messages(turns or [])
        cache_key = query.strip().lower()

        if cache_key in rag_cache:
            logger.info("RAG cache hit (stream)")
            answer = rag_cache[cache_key]
            yield {"type": "token", "content": answer}
            yield {"type": "done", "response": answer}
            return

        try:
            # The retrieval chain's first couple of streamed chunks echo
            # back input/retrieved-context metadata with no "answer" key;
            # only later chunks carry incremental answer tokens (verified
            # empirically against the real chain, not assumed).
            chunks = []
            for chunk in self.rag_chain.stream(
                {"input": query, "chat_history": chat_history},
                config=_usage_config(usage),
            ):
                token = chunk.get("answer")
                if token:
                    chunks.append(token)
                    yield {"type": "token", "content": token}

            answer = "".join(chunks)
            rag_cache[cache_key] = answer
            yield {"type": "done", "response": answer}

        except Exception:
            logger.exception("RAGAgent streaming failed")
            yield {
                "type": "done",
                "response": (
                    "❌ Unable to retrieve information from the document.\n\n"
                    "Please try again later."
                ),
            }
