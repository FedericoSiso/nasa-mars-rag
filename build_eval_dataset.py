import os
import sys
import re
import json
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "src" else CURRENT_DIR
sys.path.insert(0, CURRENT_DIR)

from ingestion import extract_and_chunk_pdfs

def extract_technical_query(text: str, idx: int) -> str:
    """Genera consultas difíciles: mitad enfocadas en acrónimos/cifras y mitad en conceptos cortos."""
    # Buscar acrónimos técnicos reales en el fragmento (ej. ISRU, EVA, GCR, SPE, MOXIE, ECLSS)
    acronyms = list(set(re.findall(r'\b[A-Z]{3,8}\b', text)))
    numbers_with_units = re.findall(r'\b\d+(?:\.\d+)?\s*(?:km|kg|mSv|days|kW|%|bar|GHz|tons)\b', text, re.IGNORECASE)
    
    # Extraer sustantivos/palabras largas clave del final del fragmento (donde suele fallar el embedding general)
    words = [w for w in re.findall(r'\b[a-zA-Z]{6,}\b', text[-350:]) if w.lower() not in {"because", "however", "through", "between", "without", "provide", "mission", "systems"}]
    key_terms = " ".join(words[:5])

    if idx % 2 == 0 and acronyms:
        acr_str = " and ".join(acronyms[:2])
        unit_hint = f" ({numbers_with_units[0]})" if numbers_with_units else ""
        return f"What is the specific operational role of {acr_str}{unit_hint} regarding {key_terms}?"
    else:
        return f"What technical constraints or design trade-offs are identified for {key_terms}?"
def generate_evaluation_dataset(num_questions: int = 10):
    pdf_folder = os.path.join(PROJECT_ROOT, "data", "raw")
    output_file = os.path.join(PROJECT_ROOT, "data", "eval_dataset.json")

    print("Extrayendo fragmentos de los PDFs descargados...")
    chunks = extract_and_chunk_pdfs(pdf_folder)
    
    valid_chunks = [c for c in chunks if len(c.page_content) > 450]
    random.seed(42)  # Semilla fija para reproducibilidad científica
    sampled_chunks = random.sample(valid_chunks, min(num_questions, len(valid_chunks)))

    eval_dataset = []

    print(f"Construyendo {len(sampled_chunks)} casos de prueba técnicos (In-Domain)...")
    for idx, chunk in enumerate(sampled_chunks):
        query = extract_technical_query(chunk.page_content, idx)
        eval_dataset.append({
            "id": f"in_domain_{idx+1}",
            "type": "in_domain",
            "question": query,
            "ground_truth": chunk.page_content[:300] + "...",
            "expected_source": chunk.metadata.get("source_doc"),
            "expected_page": chunk.metadata.get("page")
        })

    adversarial_questions = [
        {
            "id": "adversarial_1",
            "type": "out_of_domain",
            "question": "¿En qué fecha exacta alunizó el módulo lunar Eagle durante la misión Apolo 11 y cuántos kilos de roca lunar trajeron?",
            "ground_truth": "ABSTENCIÓN: Los informes técnicos disponibles no contienen información suficiente para responder a esta consulta.",
            "expected_source": None,
            "expected_page": None
        },
        {
            "id": "adversarial_2",
            "type": "out_of_domain",
            "question": "¿Cuál fue el costo total de fabricación del telescopio espacial James Webb en comparación con el Hubble?",
            "ground_truth": "ABSTENCIÓN: Los informes técnicos disponibles no contienen información suficiente para responder a esta consulta.",
            "expected_source": None,
            "expected_page": None
        }
    ]

    eval_dataset.extend(adversarial_questions)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, indent=2, ensure_ascii=False)

    print(f"¡Éxito! Se guardaron {len(sampled_chunks)} preguntas In-Domain y 2 Adversariales en '{output_file}'.")

if __name__ == "__main__":
    generate_evaluation_dataset()