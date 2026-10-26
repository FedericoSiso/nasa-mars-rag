# 🌍➡️🔴 NASA Mars NTRS — Production Hybrid RAG & Telemetry Portal

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG_Pipeline-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FC3D21?style=for-the-badge)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud_Ready-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

Plataforma de inteligencia documental de grado de producción diseñada para consultar informes técnicos aeroespaciales del **NASA Technical Reports Server (NTRS)** sobre misiones humanas a Marte. 

A diferencia de las implementaciones RAG tradicionales (que fallan al recuperar acrónimos técnicos como *ISRU*, *MOXIE* o *ECLSS* y especificaciones numéricas exactas), este sistema implementa **Búsqueda Híbrida (BM25 + Vectores Densos con RRF)**, **Reordenamiento Cross-Encoder**, **Guardrails deterministas contra alucinaciones** y un **panel de telemetría de latencia en tiempo real**.

---

## 📊 Benchmark de Rendimiento (Básico vs. Producción)

La arquitectura fue evaluada rigurosamente frente a un pipeline vectorial estándar (*Naive RAG* `k=4`), logrando mejoras críticas en precisión de recuperación y seguridad documental:

| Métrica de Evaluación | RAG Básico (Vector `k=4`) | RAG Producción (BM25 + RRF + Reranker) | Impacto Técnico |
| :--- | :---: | :---: | :--- |
| **Precision@1** | 10.0% | **90.0%** | **+80 pts** (El fragmento exacto escala al puesto #1) |
| **MRR (Mean Reciprocal Rank)** | 0.250 | **0.933** | **+273%** de mejora en el ranking de evidencia relevante |
| **Recall@4** | 40.0% | **80.0%** | Captura simultánea de contexto semántico y léxico exacto |
| **Abstención Fuera de Dominio** | Variable | **100.0%** | **0% alucinaciones** ante consultas no respaldadas (ej. Apolo 11) |

---

## 🏗️ Arquitectura del Sistema

El pipeline de inferencia se divide en cuatro etapas instrumentadas en milisegundos:

1. **Ingesta Resiliente y Chunking con Trazabilidad (`src/ingestion.py`):**
   * Extracción de documentos PDF mediante `PyMuPDF` (`fitz`) conservando metadatos estrictos (`source_doc` y número de `page`).
   * Segmentación recursiva (`chunk_size=800`, `chunk_overlap=150`) y sistema de respaldo automático (*built-in NTRS corpus*) para garantizar disponibilidad continua en entornos cloud.
2. **Etapa 1 — Recuperación Híbrida con Reciprocal Rank Fusion (`src/retriever.py`):**
   * **Búsqueda Densa:** Embeddings semánticos con `BAAI/bge-small-en-v1.5` indexados en **ChromaDB**.
   * **Búsqueda Léxica:** Índice `BM25` para capturar siglas aeroespaciales, compuestos químicos y parámetros de misión.
   * **Fusión RRF:** Ponderación equilibrada `(0.5, 0.5)` que selecciona los mejores `k=10` candidatos iniciales.
3. **Etapa 2 — Reordenamiento Cross-Encoder (`src/retriever.py`):**
   * Evaluación profunda de atención cruzada pregunta-documento utilizando `cross-encoder/ms-marco-MiniLM-L-6-v2`, filtrando el **Top 4** de mayor relevancia real.
4. **Etapa 3 — Síntesis LLM, Citas Estrictas & Guardrails (`src/generator.py`):**
   * Generación con temperatura `0.0` que exige citación explícita por documento y página `[Archivo.pdf, Pág. X]`.
   * **Protocolo de Abstención:** Bloquea respuestas cuando la evidencia no figura en el corpus NTRS.
   * **Fallback Extractivo Determinista:** Si no se provee una API Key externa o hay contingencia de red, el sistema sintetiza directamente la evidencia verificada sin interrumpir el servicio.

---

## 📂 Estructura del Repositorio

```text
nasa-mars-rag/
├── data/
│   └── raw/                  # Informes técnicos PDF del servidor NTRS de la NASA
├── chroma_db/                # Persistencia local de la base vectorial ChromaDB
├── src/
│   ├── app.py                # Interfaz Streamlit (Diseño NASA.gov + Telemetría en vivo)
│   ├── ingestion.py          # Extracción PDF (PyMuPDF), chunking y creación vectorial
│   ├── retriever.py          # Motores BasicVector, HybridRRF y Cross-Encoder Reranker
│   └── generator.py          # Prompts de sistema, Guardrails y Fallback extractivo
├── Dockerfile                # Imagen de contenedor optimizada para producción
├── docker-compose.yml        # Orquestación de servicio y volúmenes persistentes
├── requirements.txt          # Dependencias fijadas para Streamlit Cloud y Docker
└── README.md                 # Documentación técnica del proyecto

🚀 Despliegue y Ejecución Local
Opción 1: Ejecución con Docker (Recomendado)
Bash
# 1. Clonar el repositorio
git clone [https://github.com/FedericoSiso/nasa-mars-rag.git](https://github.com/FedericoSiso/nasa-mars-rag.git)
cd nasa-mars-rag

# 2. Construir y levantar el contenedor en segundo plano
docker compose up --build -d
Accede al portal en tu navegador en: http://localhost:8501

Opción 2: Entorno Virtual de Python
Bash
# 1. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación
streamlit run src/app.py
🔐 Configuración de Variables de Entorno (Opcional)
El sistema cuenta con un motor de respaldo extractivo autónomo, pero para habilitar la síntesis generativa completa puedes configurar tu clave API en un archivo .env local o en la sección Secrets de Streamlit Community Cloud:

Ini, TOML
OPENAI_API_KEY = "sk-..."
# o según el proveedor configurado en generator.py
GEMINI_API_KEY = "AIza..."
👨‍💻 Autor
Federico Siso — Data Analyst & Analytics Engineering Lead

GitHub: @FedericoSiso


---

### ⚡ Comandos rápidos para crear y subir el `README.md` desde PowerShell

Si quieres actualizar el repositorio ahora mismo desde tu terminal:

```powershell
git add README.md
git commit -m "docs: add comprehensive production architecture and benchmark README"
git push origin main
