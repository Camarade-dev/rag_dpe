import os
import warnings
# Désactiver les warnings non-critiques
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Désactiver complètement la télémétrie ChromaDB (plusieurs méthodes)
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ALLOW_RESET"] = "TRUE"

# Intercepter les erreurs de télémétrie ChromaDB (bug connu)
import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# CHANGEMENT ICI : On utilise PyMuPDF au lieu de Unstructured
from llama_index.readers.file import PyMuPDFReader 
import chromadb

# --- 1. CONFIGURATION ---
DATA_PATH = "./docs"
DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "renovation_knowledge"

def ingest_documents():
    print("🚀 Démarrage de l'ingestion (Mode PyMuPDF)...")

    # A. Modèle d'Embedding
    print("🧠 Chargement du modèle d'embedding...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model

    # B. Chunking (Découpage)
    # On garde 512/50, c'est un bon ratio pour les docs techniques
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    Settings.text_splitter = text_splitter

    # C. Configuration du lecteur PDF (Le changement clé)
    print("📂 Configuration du lecteur PyMuPDF (Rapide & Compatible)...")
    file_extractor = {
        ".pdf": PyMuPDFReader()
    }

    # D. Connexion ChromaDB
    print("💾 Connexion à ChromaDB...")
    # La télémétrie est désactivée via les variables d'environnement
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # E. Chargement
    print("⏳ Lecture des fichiers PDF...")
    try:
        documents = SimpleDirectoryReader(
            DATA_PATH, 
            recursive=True, 
            file_extractor=file_extractor
        ).load_data()
    except Exception as e:
        print(f"❌ Erreur critique lors de la lecture : {e}")
        return

    print(f"📄 {len(documents)} pages de documents chargées avec succès.")
    
    if len(documents) == 0:
        print("⚠️ Aucun document trouvé ! Vérifie le dossier ./docs")
        return

    # F. Indexation
    print("⚙️ Création des vecteurs (Embeddings)...")
    VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )

    print("✅ Ingestion Terminée ! Tous les documents sont indexés.")

if __name__ == "__main__":
    ingest_documents()