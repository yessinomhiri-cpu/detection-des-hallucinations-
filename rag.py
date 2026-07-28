"""
RAG naïf (Retrieval-Augmented Generation) 100% local, avec ChromaDB comme base vectorielle.

Pipeline:
  1. Ingestion   : lit les .txt/.pdf d'un dossier, les découpe en chunks,
                   calcule des embeddings locaux (sentence-transformers)
                   et les stocke dans une base vectorielle ChromaDB persistée sur disque.
  2. Interrogation: embedde la question, interroge ChromaDB pour récupérer
                   les k chunks les plus proches (similarité cosinus),
                   construit un prompt avec le contexte récupéré et
                   appelle un modèle local via Ollama pour générer la réponse.

Usage:
    python rag.py ingest --docs ./documents --index ./index
    python rag.py query  --index ./index --question "Ma question ?" --model llama3
"""

import argparse
import hashlib
import os
import sys
import textwrap

import chromadb
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # petit modèle local, rapide sur CPU
CHUNK_SIZE = 800          # caractères par chunk
CHUNK_OVERLAP = 150        # chevauchement entre chunks
OLLAMA_URL = "http://localhost:11434/api/generate"
COLLECTION_NAME = "documents"


# ---------------------------------------------------------------------------
# Lecture des documents
# ---------------------------------------------------------------------------

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(docs_dir: str) -> list[dict]:
    """Retourne une liste de {source, text} pour chaque fichier lu."""
    documents = []
    for root, _, files in os.walk(docs_dir):
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext == ".txt" or ext == ".md":
                    text = read_txt(path)
                elif ext == ".pdf":
                    text = read_pdf(path)
                else:
                    continue
            except Exception as e:
                print(f"[!] Impossible de lire {path}: {e}", file=sys.stderr)
                continue

            if text.strip():
                documents.append({"source": path, "text": text})
    return documents


# ---------------------------------------------------------------------------
# Découpage en chunks
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpage naïf par nombre de caractères, avec chevauchement."""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Index vectoriel (ChromaDB, persisté sur disque)
# ---------------------------------------------------------------------------

def get_collection(index_dir: str, reset: bool = False):
    """Ouvre (ou crée) la collection Chroma persistée dans index_dir."""
    os.makedirs(index_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=index_dir)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # la collection n'existait pas encore

    # cosine = similarité cosinus, plus intuitive que L2 par défaut
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def chunk_id(source: str, index: int) -> str:
    """ID stable et unique par chunk, basé sur le chemin du fichier + sa position."""
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"{digest}-{index}"


def build_index(docs_dir: str, index_dir: str) -> None:
    print(f"[*] Chargement du modèle d'embeddings '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"[*] Lecture des documents dans '{docs_dir}'...")
    documents = load_documents(docs_dir)
    if not documents:
        print("[!] Aucun document .txt/.md/.pdf trouvé. Rien à indexer.")
        return

    all_chunks, all_ids, all_metadata = [], [], []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            all_chunks.append(chunk)
            all_ids.append(chunk_id(doc["source"], i))
            all_metadata.append({"source": doc["source"]})

    print(f"[*] {len(documents)} document(s) -> {len(all_chunks)} chunk(s). Calcul des embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    # reset=True : on reconstruit la collection à chaque ingestion complète,
    # pour éviter les doublons si vous relancez 'ingest' plusieurs fois.
    collection = get_collection(index_dir, reset=True)

    # Chroma limite la taille des batchs d'insertion : on découpe par sécurité.
    batch_size = 500
    for start in range(0, len(all_chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=all_ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=all_chunks[start:end],
            metadatas=all_metadata[start:end],
        )

    print(f"[✓] Index ChromaDB sauvegardé dans '{index_dir}' ({len(all_chunks)} chunks).")


# ---------------------------------------------------------------------------
# Retrieval (recherche vectorielle via ChromaDB)
# ---------------------------------------------------------------------------

def retrieve(question: str, model, collection, top_k: int = 4):
    q_emb = model.encode([question], convert_to_numpy=True)[0].tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
    )

    retrieved = []
    for chunk, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # avec l'espace "cosine", distance = 1 - similarité cosinus
        retrieved.append({
            "chunk": chunk,
            "source": meta["source"],
            "score": 1 - distance,
        })
    return retrieved


# ---------------------------------------------------------------------------
# Génération via un modèle local (Ollama)
# ---------------------------------------------------------------------------

def build_prompt(question: str, retrieved: list[dict]) -> str:
    context = "\n\n".join(
        f"[Extrait {i+1} — source: {r['source']}]\n{r['chunk']}"
        for i, r in enumerate(retrieved)
    )
    return textwrap.dedent(f"""\
        Tu es un assistant qui répond UNIQUEMENT à partir du contexte fourni.
        Si la réponse ne se trouve pas dans le contexte, dis clairement que tu ne sais pas.

        Contexte:
        {context}

        Question: {question}

        Réponse:""")


def generate_answer(prompt: str, model_name: str = "llama3") -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def query(index_dir: str, question: str, model_name: str, top_k: int) -> None:
    print("[*] Chargement de l'index ChromaDB et du modèle d'embeddings...")
    collection = get_collection(index_dir)
    if collection.count() == 0:
        print(f"[!] L'index dans '{index_dir}' est vide. Lancez d'abord la commande 'ingest'.")
        return
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    retrieved = retrieve(question, embed_model, collection, top_k=top_k)

    print("\n[*] Extraits récupérés :")
    for r in retrieved:
        print(f"  - score={r['score']:.3f}  source={r['source']}")

    prompt = build_prompt(question, retrieved)

    print(f"\n[*] Génération de la réponse avec Ollama (modèle='{model_name}')...")
    try:
        answer = generate_answer(prompt, model_name)
    except requests.exceptions.ConnectionError:
        print(
            "[!] Impossible de contacter Ollama sur http://localhost:11434.\n"
            "    Vérifiez qu'Ollama est lancé (`ollama serve`) et que le modèle "
            f"'{model_name}' est disponible (`ollama pull {model_name}`)."
        )
        return

    print("\n=== Réponse ===")
    print(answer)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG naïf 100% local (embeddings locaux + Ollama).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Indexer les documents d'un dossier")
    p_ingest.add_argument("--docs", default="./documents", help="Dossier contenant les .txt/.md/.pdf")
    p_ingest.add_argument("--index", default="./index", help="Dossier où sauvegarder l'index")

    p_query = sub.add_parser("query", help="Poser une question au RAG")
    p_query.add_argument("--index", default="./index", help="Dossier de l'index à charger")
    p_query.add_argument("--question", required=True, help="Question à poser")
    p_query.add_argument("--model", default="llama3", help="Nom du modèle Ollama à utiliser")
    p_query.add_argument("--top-k", type=int, default=4, help="Nombre de chunks à récupérer")

    args = parser.parse_args()

    if args.command == "ingest":
        build_index(args.docs, args.index)
    elif args.command == "query":
        query(args.index, args.question, args.model, args.top_k)


if __name__ == "__main__":
    main()
