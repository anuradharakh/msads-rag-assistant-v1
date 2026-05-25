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


def answer_question(question: str):
    config, retriever, reranker, generator, router = load_components()

    route, confidence = router.route(question)

    if route != "MSADS_QUERY":
        return direct_response(route), [], route, confidence

    fetch_k = config["retrieval"]["fetch_k"]
    top_k = config["retrieval"]["top_k"]

    chunks = retriever.retrieve(
        query=question,
        fetch_k=fetch_k,
        top_k=fetch_k,
    )

    chunks = reranker.rerank(
        query=question,
        chunks=chunks,
        top_n=top_k,
    )

    answer = generator.generate(
        question=question,
        chunks=chunks[: config["generation"]["max_context_chunks"]],
    )

    return answer, chunks, route, confidence


def render_sources(chunks):
    if not chunks:
        return

    with st.expander("Retrieved Sources"):
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})

            st.markdown(
                f"""
### Source {index}

**Title:** {metadata.get("title", "Unknown")}

**Section:** {metadata.get("section", "Unknown")}

**URL:** {metadata.get("url", "Unknown")}

**Chunk Type:** {metadata.get("chunk_type", "Unknown")}
"""
            )


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

    st.title(config["ui"]["title"])
    st.caption(
        "Ask questions about the University of Chicago MS in Applied Data Science program."
    )

    render_chat_history()

    question = st.chat_input(
        "Ask about admissions, curriculum, online program, capstone, tuition, CPT/OPT, or career outcomes..."
    )

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
        with st.spinner("Thinking..."):
            answer, chunks, route, confidence = answer_question(question)

        st.markdown(answer)

        if route == "MSADS_QUERY":
            st.caption(f"Intent: {route} | Confidence: {confidence:.2f}")
            render_sources(chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": chunks,
        }
    )


if __name__ == "__main__":
    main()