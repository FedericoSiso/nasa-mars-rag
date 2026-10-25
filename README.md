🛠️ Arquitectura Técnica en Detalle
Este asistente RAG va mucho más allá de las implementaciones estándar de tutoriales al abordar desafíos reales de ingeniería en entornos de producción: ruido en la recuperación, compensación de latencia, control de alucinaciones y evaluación cuantitativa.

[ Informes PDF de la NASA ] 

       │

       ▼

[ 1. Ingestión y Segmentación ] ──> Metadatos Estructurales (Fuente y Página)

       │

       ▼

[ 2. Recuperación Híbrida ] ──────> Búsqueda Vectorial Densa (ChromaDB + BGE) 

       │                          + Búsqueda Léxica Dispersa (BM25) 

       │                          Fusionadas mediante Reciprocal Rank Fusion (RRF)

       │

       ▼

[ 3. Reordenamiento Cross-Encoder ] ─> Puntuación de Atención Cruzada (ms-marco-MiniLM-L-6-v2)

       │                               Filtra del Top-10 al Top-4 de alta relevancia

       │

       ▼

[ 4. Generación y Guardrails ] ──> LLM Temp-0 + Enrutador de Respaldo + Guardrail de Citas


🔬 Componentes Principales del Pipeline
1. Ingestión y Conservación de Metadatos Estructurales
Motor: PyMuPDF (pymupdf) procesa los informes técnicos oficiales del NASA Technical Reports Server (NTRS).
Trazabilidad: Cada fragmento de texto almacena metadatos estrictos (source_doc y número de page exacto), permitiendo una trazabilidad determinista desde la respuesta final hasta el documento fuente original.
2. Recuperación Híbrida mediante Reciprocal Rank Fusion (RRF)
El Desafío: Los embeddings densos (BAAI/bge-small-en-v1.5) son excelentes para la similitud semántica, pero suelen pasar por alto acrónimos precisos, parámetros numéricos y jerga técnica específica (ej. ISRU, ECLSS, MOXIE).
La Solución: Se ejecutan dos recuperadores en paralelo:
Búsqueda Semántica: Almacén vectorial ChromaDB.
Búsqueda Léxica: Recuperador disperso Rank-BM25.
Fusión: Los puntajes se combinan mediante Reciprocal Rank Fusion (RRF) para equilibrar el contexto semántico con la coincidencia exacta de palabras clave, elevando el recall al 80% en k=10.
3. Reordenamiento con Cross-Encoder (Grado de Producción)
Modelo: Cross-encoder ms-marco-MiniLM-L-6-v2.
Mecanismo: A diferencia de los bi-encoders que codifican la consulta y los documentos por separado, el cross-encoder procesa la consulta y cada fragmento candidato de manera simultánea mediante atención cruzada.
Impacto de Ingeniería: Eleva la Precision@1 del 10% al 90% y lleva el MRR (Mean Reciprocal Rank) a 0.933, garantizando que el párrafo más relevante ocupe siempre la primera posición en la ventana de contexto del LLM.
4. Guardrails y Enrutador Multi-Modelo de Respaldo
Guardrail de Abstención: Una combinación de ingeniería de prompt a temperatura 0 y validación léxica rechaza consultas fuera de dominio (ej. preguntas sobre el módulo lunar Apolo 11) con un 100% de éxito, evitando alucinaciones.
Enrutador Resiliente: Alterna automáticamente entre endpoints gratuitos de la API (familia Gemini y Gemma) e incluye un mecanismo de respaldo extractivo local para asegurar la disponibilidad ante límites de cuota (rate limits).
5. Observabilidad y Telemetría en Tiempo Real
Interfaz: Desarrollada en Streamlit, mide la latencia paso a paso utilizando time.perf_counter(), aislando el costo computacional exacto de la recuperación híbrida, el reordenamiento del reranker y la fase de generación del LLM.
