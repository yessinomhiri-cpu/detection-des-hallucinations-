# RAG naïf — 100% local (avec ChromaDB)

Un pipeline RAG (Retrieval-Augmented Generation) simple et pédagogique, qui tourne
entièrement en local :

- **Embeddings** : `sentence-transformers` (modèle `all-MiniLM-L6-v2`, tourne sur CPU)
- **Base vectorielle** : [ChromaDB](https://www.trychroma.com/), persistée sur disque
  (recherche par similarité cosinus via un index HNSW)
- **Génération** : un LLM local via [Ollama](https://ollama.com)
- **Documents supportés** : `.txt`, `.md`, `.pdf`

Le pipeline reste "naïf" au sens où il n'y a pas de re-ranking, de filtrage avancé
ou de découpage sémantique — mais le retrieval lui-même s'appuie désormais sur une
vraie base vectorielle plutôt que sur une recherche brute force en numpy.

## Installation

```bash
# 1. Installer les dépendances Python
pip install -r requirements.txt

# 2. Installer Ollama (si ce n'est pas déjà fait) : https://ollama.com/download
# 3. Récupérer un modèle local, par exemple llama3
ollama pull llama3

# 4. Démarrer Ollama (souvent lancé automatiquement, sinon :)
ollama serve
```

## Utilisation

### 1. Déposer vos documents

Placez vos fichiers `.txt`, `.md` ou `.pdf` dans le dossier `documents/`
(ou n'importe quel autre dossier de votre choix).

### 2. Indexer les documents

```bash
python rag.py ingest --docs ./documents --index ./index
```

Cela va :
1. lire tous les fichiers du dossier `documents/`
2. les découper en chunks de ~800 caractères (avec 150 caractères de chevauchement)
3. calculer un embedding pour chaque chunk
4. sauvegarder les embeddings + le texte des chunks dans `./index`

### 3. Poser une question

```bash
python rag.py query --index ./index --question "Quelle est la politique de remboursement ?" --model llama3
```

Le script va :
1. embedder votre question
2. retrouver les chunks les plus proches (similarité cosinus)
3. construire un prompt avec ces extraits comme contexte
4. envoyer ce prompt à Ollama et afficher la réponse générée

Options utiles :
- `--top-k 6` pour récupérer plus (ou moins) de chunks de contexte
- `--model mistral` (ou tout autre modèle installé via `ollama pull ...`)

## Structure du projet

```
naive-rag/
├── rag.py              # tout le pipeline (ingest + query)
├── requirements.txt
├── documents/          # vos fichiers sources (.txt / .md / .pdf)
└── index/              # base ChromaDB persistée (généré automatiquement)
```

## Interface Streamlit

Une interface web est disponible en plus de la ligne de commande :

```bash
streamlit run streamlit_app.py
```

Elle s'ouvre automatiquement dans votre navigateur (`http://localhost:8501`) et permet de :
- choisir le dossier de documents, le dossier d'index, le modèle Ollama et le top-k depuis la barre latérale
- lancer l'indexation d'un clic (bouton "Indexer les documents")
- discuter avec le RAG dans une interface de type chat, avec la réponse qui s'affiche en streaming
- consulter les sources utilisées pour chaque réponse (fichier, score de similarité, extrait du chunk)

## Note sur la ré-ingestion

À chaque `ingest`, la collection ChromaDB est entièrement reconstruite (les anciens
chunks sont supprimés) pour éviter les doublons. Si vous ajoutez un document, il
suffit de relancer `ingest` sur le dossier complet.

## Limites de ce RAG "naïf" (pistes d'amélioration)

- Découpage par nombre de caractères fixes → un découpage par phrases/paragraphes
  ou "sémantique" donnerait de meilleurs chunks.
- Pas de re-ranking, pas de filtrage par métadonnées avancé.
- Pas de citation précise (numéro de page, position exacte) au-delà du fichier source.
- Un seul type de similarité (cosinus) et un seul modèle d'embeddings.
- Ré-ingestion complète à chaque fois plutôt qu'une mise à jour incrémentale.

Ces limites sont volontaires : ce projet sert de base claire pour comprendre le
mécanisme avant d'aller vers une architecture RAG plus avancée.