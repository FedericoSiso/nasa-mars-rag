from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class BasicVectorRetriever:
    """Retriever 1: Búsqueda puramente vectorial en ChromaDB."""
    def __init__(self, vectorstore, k: int = 4):
        self.vectorstore = vectorstore
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)


class HybridRRFRetriever:
    """Retriever 2: Búsqueda Híbrida (BM25 + Vector Search) usando Reciprocal Rank Fusion (RRF)."""
    def __init__(self, vectorstore, chunks: list[Document], k: int = 10, weights=(0.5, 0.5)):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.k = k
        self.w_bm25, self.w_vec = weights

        # Tokenización ligera para BM25 nativo
        tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def invoke(self, query: str) -> list[Document]:
        # 1. Candidatos Léxicos (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.k]
        bm25_docs = [self.chunks[i] for i in top_bm25_idx]

        # 2. Candidatos Semánticos (Chroma Vector Search)
        vec_docs = self.vectorstore.similarity_search(query, k=self.k)

        # 3. Fusión de Rango Recíproco (RRF): Score = w / (60 + rank)
        rrf_scores = {}
        doc_map = {}

        for rank, doc in enumerate(bm25_docs):
            key = (doc.metadata.get("source_doc"), doc.metadata.get("page"), doc.page_content[:80])
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.w_bm25 / (60 + rank + 1)

        for rank, doc in enumerate(vec_docs):
            key = (doc.metadata.get("source_doc"), doc.metadata.get("page"), doc.page_content[:80])
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.w_vec / (60 + rank + 1)

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:self.k]]


class ProductionRerankRetriever:
    """Retriever 3 (Producción): Híbrido RRF (Top-10) + Cross-Encoder Reranker (Top-4)."""
    def __init__(self, hybrid_retriever: HybridRRFRetriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 4):
        self.hybrid_retriever = hybrid_retriever
        self.cross_encoder = CrossEncoder(model_name)
        self.top_n = top_n

    def invoke(self, query: str) -> list[Document]:
        candidates = self.hybrid_retriever.invoke(query)
        return self.rerank(query, candidates)

    def rerank(self, query: str, candidates: list[Document]) -> list[Document]:
        if not candidates:
            return []
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        scored_docs = []
        for doc, score in zip(candidates, scores):
            # Guardamos copia con el score en metadata para mostrarlo en Streamlit
            new_meta = dict(doc.metadata)
            new_meta["relevance_score"] = float(score)
            scored_docs.append(
                (float(score), Document(page_content=doc.page_content, metadata=new_meta))
            )

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:self.top_n]]
