import os
import sys
import json
import time
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "src" else CURRENT_DIR
sys.path.insert(0, CURRENT_DIR)

from ingestion import extract_and_chunk_pdfs, build_vector_store
from generator import answer_query
from retriever import BasicVectorRetriever, HybridRRFRetriever, ProductionRerankRetriever

def run_architecture_comparison():
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    eval_file = os.path.join(PROJECT_ROOT, "data", "eval_dataset.json")
    chroma_dir = os.path.join(PROJECT_ROOT, "chroma_db")

    print("1. Cargando fragmentos y ChromaDB...")
    chunks = extract_and_chunk_pdfs(raw_dir)
    vectorstore = build_vector_store(chunks, persist_dir=chroma_dir)

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print("2. Inicializando las 3 arquitecturas...")
    retriever_basic = BasicVectorRetriever(vectorstore, k=4)
    retriever_hybrid = HybridRRFRetriever(vectorstore, chunks, k=4, weights=(0.5, 0.5))
    
    hybrid_k10 = HybridRRFRetriever(vectorstore, chunks, k=10, weights=(0.5, 0.5))
    retriever_prod = ProductionRerankRetriever(hybrid_k10, top_n=4)

    configs = {
        "1. RAG Básico (Vector k=4)": retriever_basic,
        "2. RAG Híbrido (BM25 + Vector RRF)": retriever_hybrid,
        "3. Híbrido + Cross-Encoder Reranker": retriever_prod
    }

    summary_rows = []

    for name, retriever in configs.items():
        print(f"\n-> Evaluando: {name}...")
        retrieval_latencies = []
        llm_latencies = []
        top1_page_hits = 0
        top4_page_hits = 0
        reciprocal_ranks = []
        in_domain_total = 0
        abstention_hits = 0
        out_domain_total = 0

        for item in eval_data:
            if item["type"] == "in_domain":
                in_domain_total += 1
                t0 = time.perf_counter()
                docs = retriever.invoke(item["question"])
                elapsed_ret_ms = (time.perf_counter() - t0) * 1000
                retrieval_latencies.append(elapsed_ret_ms)

                # Evaluar posición exacta de (Documento + Página)
                hit_rank = 0
                for rank_idx, d in enumerate(docs):
                    if (d.metadata.get("source_doc") == item["expected_source"] and 
                        d.metadata.get("page") == item["expected_page"]):
                        hit_rank = rank_idx + 1
                        break
                
                if hit_rank == 1:
                    top1_page_hits += 1
                if hit_rank > 0:
                    top4_page_hits += 1
                    reciprocal_ranks.append(1.0 / hit_rank)
                else:
                    reciprocal_ranks.append(0.0)
            else:
                out_domain_total += 1
                print(f"   [LLM Call {out_domain_total}/2] Probando Guardrail de Abstención...")
                t0 = time.perf_counter()
                res = answer_query(item["question"], retriever)
                elapsed_total_ms = (time.perf_counter() - t0) * 1000
                llm_latencies.append(elapsed_total_ms)

                if "ABSTENCIÓN:" in res["answer"]:
                    abstention_hits += 1

        avg_ret_ms = sum(retrieval_latencies) / max(1, len(retrieval_latencies))
        avg_total_ms = sum(llm_latencies) / max(1, len(llm_latencies))
        mrr_score = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))

        summary_rows.append({
            "Arquitectura": name,
            "Precision@1 Pág. Exacta (%)": round((top1_page_hits / max(1, in_domain_total)) * 100, 1),
            "Recall@4 Pág. Exacta (%)": round((top4_page_hits / max(1, in_domain_total)) * 100, 1),
            "MRR (Ranking)": round(mrr_score, 3),
            "Abstención (%)": round((abstention_hits / max(1, out_domain_total)) * 100, 1),
            "Retrieval+Rerank (ms)": round(avg_ret_ms, 1),
            "Total c/ LLM (ms)": round(avg_total_ms, 0)
        })

    df_results = pd.DataFrame(summary_rows)
    print("\n=== TABLA DE RESULTADOS PARA TU README.MD ===\n")
    try:
        print(df_results.to_markdown(index=False))
    except ImportError:
        print(df_results.to_string(index=False))

if __name__ == "__main__":
    run_architecture_comparison()