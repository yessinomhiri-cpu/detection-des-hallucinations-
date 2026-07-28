"""
Interface Streamlit pour le RAG naïf (ChromaDB + Ollama).

Lancement :
    streamlit run streamlit_app.py
"""

import os
import time

import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

from rag import (
    EMBEDDING_MODEL_NAME,
    OLLAMA_URL,
    build_index,
    build_prompt,
    get_collection,
    retrieve,
)

st.set_page_config(page_title="RAG naïf local", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# Ressources mises en cache (évite de recharger le modèle à chaque interaction)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_ollama_models() -> list[str]:
    """Récupère la liste des modèles Ollama installés localement."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def stream_ollama_answer(prompt: str, model_name: str):
    """Génère la réponse en streaming (yield token par token) depuis Ollama."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": model_name, "prompt": prompt, "stream": True},
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        import json as _json
        chunk = _json.loads(line)
        yield chunk.get("response", "")
        if chunk.get("done"):
            break


# ---------------------------------------------------------------------------
# Barre latérale : configuration + ingestion
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")

    docs_dir = st.text_input("Dossier des documents", value="./documents")
    index_dir = st.text_input("Dossier de l'index", value="./index")

    st.divider()

    available_models = get_ollama_models()
    if available_models:
        model_name = st.selectbox("Modèle Ollama", available_models)
    else:
        st.warning("Ollama ne semble pas accessible sur localhost:11434.")
        model_name = st.text_input("Modèle Ollama", value="llama3")

    top_k = st.slider("Nombre de chunks récupérés (top-k)", min_value=1, max_value=10, value=4)

    st.divider()

    st.subheader("📥 Indexation")
    st.caption(
        "Reconstruit entièrement l'index à partir des documents du dossier ci-dessus. "
        "À relancer si vous ajoutez ou modifiez des fichiers."
    )
    if st.button("Indexer les documents", use_container_width=True):
        if not os.path.isdir(docs_dir):
            st.error(f"Le dossier '{docs_dir}' n'existe pas.")
        else:
            with st.spinner("Indexation en cours... (peut prendre quelques minutes)"):
                try:
                    build_index(docs_dir, index_dir)
                    st.success("Index reconstruit avec succès !")
                except Exception as e:
                    st.error(f"Erreur pendant l'indexation : {e}")

    # État de l'index
    try:
        collection = get_collection(index_dir)
        count = collection.count()
        if count > 0:
            st.info(f"📊 Index actuel : {count} chunks indexés.")
        else:
            st.warning("L'index est vide. Cliquez sur 'Indexer les documents'.")
    except Exception:
        st.warning("Aucun index trouvé pour le moment.")


# ---------------------------------------------------------------------------
# Corps principal : chat
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Affiche l'historique de conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources utilisées"):
                for src in msg["sources"]:
                    st.markdown(f"**{os.path.basename(src['source'])}** — score : {src['score']:.3f}")
                    st.text(src["chunk"][:400] + ("..." if len(src["chunk"]) > 400 else ""))
                    st.divider()

question = st.chat_input("Posez votre question sur les documents indexés...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            collection = get_collection(index_dir)
        except Exception as e:
            st.error(f"Impossible d'ouvrir l'index dans '{index_dir}' : {e}")
            st.stop()

        if collection.count() == 0:
            st.warning("L'index est vide. Indexez d'abord vos documents depuis la barre latérale.")
            st.stop()

        with st.spinner("Recherche des passages pertinents..."):
            embed_model = load_embedding_model()
            retrieved = retrieve(question, embed_model, collection, top_k=top_k)
            prompt = build_prompt(question, retrieved)

        with st.expander("📎 Sources utilisées", expanded=False):
            for r in retrieved:
                st.markdown(f"**{os.path.basename(r['source'])}** — score : {r['score']:.3f}")
                st.text(r["chunk"][:400] + ("..." if len(r["chunk"]) > 400 else ""))
                st.divider()

        answer_placeholder = st.empty()
        full_answer = ""
        try:
            for token in stream_ollama_answer(prompt, model_name):
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")
                time.sleep(0.005)
            answer_placeholder.markdown(full_answer)
        except requests.exceptions.ConnectionError:
            full_answer = (
                "❌ Impossible de contacter Ollama sur `http://localhost:11434`.\n\n"
                "Vérifiez qu'Ollama est lancé (`ollama serve`) et que le modèle "
                f"`{model_name}` est installé (`ollama pull {model_name}`)."
            )
            answer_placeholder.error(full_answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": retrieved,
    })