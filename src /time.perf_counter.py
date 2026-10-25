import time

def profile_rag_pipeline(query: str, hybrid_retriever, reranker_Compressor, llm_chain):
    # 1. Medir tiempo de recuperación híbrida (BM25 + Vector)
    t0 = time.perf_counter()
    initial_docs = hybrid_retriever.invoke(query)
    t_retrieval = (time.perf_counter() - t0) * 1000  # en ms

    # 2. Medir tiempo exclusivo del Cross-Encoder Reranker
    t1 = time.perf_counter()
    reranked_docs = reranker_Compressor.compress_documents(initial_docs, query)
    t_rerank = (time.perf_counter() - t1) * 1000    # en ms

    # 3. Medir tiempo de generación del LLM
    t2 = time.perf_counter()
    context_str = "\n\n".join([d.page_content for d in reranked_docs])
    response = llm_chain.invoke({"context": context_str, "question": query})
    t_llm = (time.perf_counter() - t2) * 1000       # en ms

    return {
        "answer": response.content,
        "latency_breakdown_ms": {
            "hybrid_retrieval_ms": round(t_retrieval, 1),
            "reranker_ms": round(t_rerank, 1),
            "llm_generation_ms": round(t_llm, 1),
            "total_ms": round(t_retrieval + t_rerank + t_llm, 1)
        }
    }
