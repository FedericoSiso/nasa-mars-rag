import os
import json
import time
import requests

NASA_API_SEARCH_URL = "https://ntrs.nasa.gov/api/citations/search"
NASA_BASE_URL = "https://ntrs.nasa.gov"

def download_mars_reports(output_dir: str = "./data/raw", max_pdfs: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    
    # Consulta enfocada en los retos reales de llevar humanos a Marte
    payload = {
        "q": "human Mars mission radiation life support ISRU",
        "page": {"size": 40, "from": 0}
    }
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "MarsRAGPortfolioProject/1.0"
    }

    print("Consultando la API pública de NASA NTRS...")
    response = requests.post(NASA_API_SEARCH_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    results = response.json().get("results", [])

    downloaded_catalog = []
    count = 0

    for item in results:
        if count >= max_pdfs:
            break

        c_id = item.get("id")
        title = item.get("title", "Untitled_Report")
        pub_date = item.get("publications", [{}])[0].get("publicationDate", "Unknown_Date")[:4]
        downloads = item.get("downloads", [])

        # Buscar si el registro tiene un archivo PDF directo disponible
        pdf_link = None
        for d in downloads:
            links = d.get("links", {})
            if "pdf" in links:
                pdf_link = NASA_BASE_URL + links["pdf"]
                break

        if pdf_link:
            filename = f"NASA_{c_id}_{pub_date}.pdf"
            filepath = os.path.join(output_dir, filename)

            try:
                print(f"[{count + 1}/{max_pdfs}] Descargando: {title[:60]}...")
                pdf_resp = requests.get(pdf_link, headers=headers, timeout=60)
                
                # Verificar que realmente sea un PDF válido
                if pdf_resp.status_code == 200 and b"%PDF" in pdf_resp.content[:1024]:
                    with open(filepath, "wb") as f:
                        f.write(pdf_resp.content)
                    
                    downloaded_catalog.append({
                        "nasa_id": c_id,
                        "filename": filename,
                        "title": title,
                        "year": pub_date,
                        "pdf_url": pdf_link
                    })
                    count += 1
                    time.sleep(1.0)  # Respetar rate-limits de la API de la NASA
            except Exception as e:
                print(f"  -> Error descargando {c_id}: {e}")

    # Guardar el catálogo para documentar en tu README de GitHub qué informes usaste
    catalog_path = os.path.join(output_dir, "metadata_catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(downloaded_catalog, f, indent=2, ensure_ascii=False)

    print(f"\n¡Listo! Se descargaron {count} informes en '{output_dir}'.")
    print(f"Catálogo guardado en '{catalog_path}'.")

if __name__ == "__main__":
    download_mars_reports()