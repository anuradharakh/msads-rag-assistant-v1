import streamlit as st

st.set_page_config(
    page_title="MSADS RAG Assistant",
    page_icon="🎓",
    layout="wide",
)

from src.generation.generator import AnswerGenerator
from src.indexing.embedding import EmbeddingModel
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.config_loader import load_config
from src.utils.query_router import QueryRouter, direct_response


SAMPLE_QUESTIONS = [
    "What are the core courses in the MSADS curriculum?",
    "List names of all courses offered in the program.",
    "Does the MSADS program support CPT and OPT?",
    "What are the admission requirements?",
    "What is the difference between online and in-person programs?",
    "What career outcomes do graduates typically achieve?",
]


@st.cache_resource
def load_components():
    config = load_config("config.yaml")
    embedding_config = config["models"]["embedding"]

    embedding_model = EmbeddingModel(
        model_name=embedding_config["name"],
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
    )

    dense_retriever = DenseRetriever(
        index_dir=config["paths"]["index_dir"],
        embedding_model=embedding_model,
    )

    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        index_dir=config["paths"]["index_dir"],
        rrf_k=config["retrieval"]["hybrid"]["rrf_k"],
    )

    reranker = CrossEncoderReranker(
        model_name=config["models"]["reranker"]["name"],
    )

    generator = AnswerGenerator(
        llm_config=config["models"]["llm"],
        generation_config=config["generation"],
    )

    router = QueryRouter(
        model_name=embedding_config["name"],
        threshold=0.55,
    )

    return config, retriever, reranker, generator, router


def render_sidebar(config):
    st.sidebar.title("🎓 MSADS Assistant")
    st.sidebar.caption("Production-style RAG chatbot")

    st.sidebar.markdown("### Architecture")
    st.sidebar.write("Hybrid BM25 + Dense Retrieval")
    st.sidebar.write("RRF Fusion + Cross-Encoder Reranker")
    st.sidebar.write("Grounded GPT Answer Generation")

    st.sidebar.markdown("### Models")
    st.sidebar.write(f"Embedding: `{config['models']['embedding']['name']}`")
    st.sidebar.write(f"Reranker: `{config['models']['reranker']['name']}`")
    st.sidebar.write(f"LLM: `{config['models']['llm']['name']}`")

    st.sidebar.markdown("### Evaluation")
    st.sidebar.metric("Hit Rate@5", "1.00")
    st.sidebar.caption("Based on local evaluation set")


def answer_question(question: str):
    config, retriever, reranker, generator, router = load_components()

    route, confidence = router.route(question)

    if route != "MSADS_QUERY":
        return direct_response(route), [], route, confidence

    status = st.status("Running RAG pipeline...", expanded=True)
    status.write("✅ Intent routed as MSADS question")
    status.write("🔎 Running hybrid dense + BM25 retrieval")

    chunks = retriever.retrieve(
        query=question,
        fetch_k=config["retrieval"]["fetch_k"],
        top_k=config["retrieval"]["fetch_k"],
    )

    status.write("⚖️ Applying cross-encoder reranking")

    chunks = reranker.rerank(
        query=question,
        chunks=chunks,
        top_n=config["retrieval"]["top_k"],
    )

    status.write("🧠 Generating grounded answer")
    status.update(label="RAG pipeline completed", state="complete", expanded=False)

    answer = generator.generate(
        question=question,
        chunks=chunks[: config["generation"]["max_context_chunks"]],
    )

    return answer, chunks, route, confidence


def render_sources(chunks):
    if not chunks:
        return

    with st.expander("🔍 Retrieved Sources", expanded=False):
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            score = chunk.get("reranker_score", chunk.get("score", 0.0))
            preview = chunk.get("chunk_text", "")[:700]

            st.markdown(
                f"""
#### Source {index}

**Title:** {metadata.get("title", "Unknown")}  
**Section:** {metadata.get("section", "Unknown")}  
**Chunk Type:** `{metadata.get("chunk_type", "header_recursive")}`  
**Score:** `{score:.4f}`  
**URL:** {metadata.get("url", "Unknown")}

**Preview:**  
{preview}

---
"""
            )


def render_suggested_questions():
    st.markdown("#### Suggested follow-up questions")

    suggestions = [
        "What electives are available?",
        "What is the capstone project?",
        "Can international students apply?",
        "Is the online program visa eligible?",
    ]

    cols = st.columns(2)

    for index, question in enumerate(suggestions):
        with cols[index % 2]:
            st.caption(f"• {question}")


def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                render_sources(message["sources"])


def main():
    initialize_chat()

    config, _, _, _, _ = load_components()
    render_sidebar(config)

    st.title(config["ui"]["title"])
    st.caption(
        "Ask questions about admissions, curriculum, CPT/OPT, capstone, online options, and career outcomes."
    )

    with st.expander("✨ Try sample questions", expanded=True):
        cols = st.columns(2)

        for index, sample in enumerate(SAMPLE_QUESTIONS):
            with cols[index % 2]:
                if st.button(sample, key=f"sample_{index}"):
                    st.session_state.pending_question = sample

    render_chat_history()

    question = st.chat_input("Ask an MSADS question...")

    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer, chunks, route, confidence = answer_question(question)

        st.markdown(answer)
        st.caption(f"Intent: `{route}` | Confidence: `{confidence:.2f}`")

        render_sources(chunks)

        if route == "MSADS_QUERY":
            render_suggested_questions()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": chunks,
        }
    )


if __name__ == "__main__":
    main()