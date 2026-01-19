"""
Script d'ingestion compatible avec Render
Utilise les mêmes embeddings API que l'API RAG (pas de torch/sentence-transformers)
"""
import os
import sys
import warnings
import time

# Désactiver les warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ALLOW_RESET"] = "TRUE"

import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import PyMuPDFReader
import chromadb
from typing import List
import asyncio
import requests

# Ajouter le chemin parent pour importer le wrapper d'embeddings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.getenv("DOCS_PATH", os.path.join(BASE_DIR, "docs"))
DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "data", "chroma_db"))
COLLECTION_NAME = "renovation_knowledge"

# Wrapper d'embeddings compatible avec l'API Hugging Face
from llama_index.core.embeddings import BaseEmbedding

class HuggingFaceAPIEmbeddingForIngestion(BaseEmbedding):
    """
    Wrapper pour utiliser l'API Hugging Face embeddings via router.huggingface.co
    Compatible avec l'ingestion (batch processing)
    """
    def __init__(self, api_key: str, model_name: str = "intfloat/multilingual-e5-base"):
        super().__init__(model_name=model_name)
        object.__setattr__(self, 'api_key', api_key)
        object.__setattr__(self, 'url', f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction")
        object.__setattr__(self, 'headers', {"Authorization": f"Bearer {api_key}"})
        object.__setattr__(self, '_request_count', 0)
        object.__setattr__(self, '_last_request_time', 0)
    
    def _rate_limit(self):
        """Simple rate limiting pour éviter de surcharger l'API"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < 0.1:  # Max 10 requêtes/seconde
            time.sleep(0.1 - time_since_last)
        object.__setattr__(self, '_last_request_time', time.time())
        object.__setattr__(self, '_request_count', self._request_count + 1)
    
    def _get_embedding_single(self, text: str, prefix: str = "passage") -> List[float]:
        """Obtenir l'embedding d'un texte unique"""
        self._rate_limit()
        
        # Ajouter le préfixe pour multilingual-e5-base
        if not text.strip().startswith("query:") and not text.strip().startswith("passage:"):
            text = f"{prefix}: {text}"
        
        payload = {"inputs": text}
        
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 503:
                # Modèle en cours de chargement, attendre et réessayer
                print("⏳ Modèle en cours de chargement, attente...")
                time.sleep(20)
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
            
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                embedding = data[0] if isinstance(data[0], list) else data
                return [float(x) for x in embedding]
            raise ValueError(f"Format de réponse inattendu: {type(data)}")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Erreur API embedding: {e}")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding_single(query, prefix="query")
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding_single(text, prefix="passage")
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._aget_query_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Obtenir les embeddings de plusieurs textes (batch)"""
        embeddings = []
        total = len(texts)
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"   📊 Embedding {i+1}/{total}...")
            embeddings.append(self._get_text_embedding(text))
        return embeddings
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)


def check_collection_empty(db_path: str, collection_name: str) -> bool:
    """Vérifie si la collection ChromaDB est vide ou n'existe pas"""
    try:
        if not os.path.exists(db_path):
            return True
        
        db = chromadb.PersistentClient(path=db_path)
        try:
            collection = db.get_collection(collection_name)
            count = collection.count()
            print(f"📊 Collection '{collection_name}' contient {count} documents")
            return count == 0
        except Exception:
            return True
    except Exception as e:
        print(f"⚠️ Erreur lors de la vérification : {e}")
        return True


def ingest_documents(force: bool = False, max_docs: int = None):
    """
    Indexe les documents PDF dans ChromaDB
    
    Args:
        force: Si True, réindexe même si des documents existent déjà
        max_docs: Limite le nombre de documents à indexer (utile pour les tests)
    """
    print("=" * 60)
    print("🚀 INGESTION DES DOCUMENTS (Mode API Hugging Face)")
    print("=" * 60)
    
    # Vérifier les variables d'environnement
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("❌ HUGGINGFACE_API_KEY non définie !")
        print("💡 Configurez-la avec : export HUGGINGFACE_API_KEY=hf_xxx")
        return False
    
    print(f"📂 Dossier documents : {DATA_PATH}")
    print(f"💾 Base ChromaDB : {DB_PATH}")
    print(f"🔑 API Key : {api_key[:10]}...{api_key[-4:]}")
    
    # Vérifier si l'ingestion est nécessaire
    if not force and not check_collection_empty(DB_PATH, COLLECTION_NAME):
        print("✅ La collection contient déjà des documents. Utilisez force=True pour réindexer.")
        return True
    
    # Vérifier que le dossier docs existe
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dossier documents introuvable : {DATA_PATH}")
        return False
    
    # A. Modèle d'Embedding via API
    print("\n🧠 Initialisation des embeddings via API Hugging Face...")
    try:
        embed_model = HuggingFaceAPIEmbeddingForIngestion(
            api_key=api_key,
            model_name="intfloat/multilingual-e5-base"  # Même modèle que l'API
        )
        Settings.embed_model = embed_model
        print("✅ Embeddings initialisés")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation des embeddings : {e}")
        return False
    
    # B. Chunking
    print("\n📏 Configuration du chunking...")
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    Settings.text_splitter = text_splitter
    print("✅ Chunking configuré (512 tokens, overlap 50)")
    
    # C. Lecteur PDF
    print("\n📂 Configuration du lecteur PDF...")
    file_extractor = {".pdf": PyMuPDFReader()}
    
    # D. Connexion ChromaDB
    print("\n💾 Connexion à ChromaDB...")
    os.makedirs(DB_PATH, exist_ok=True)
    db = chromadb.PersistentClient(path=DB_PATH)
    
    # Supprimer la collection existante si force
    if force:
        try:
            db.delete_collection(COLLECTION_NAME)
            print("🗑️ Collection existante supprimée")
        except Exception:
            pass
    
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME, embedding_function=None)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("✅ ChromaDB connecté")
    
    # E. Chargement des documents
    print("\n⏳ Lecture des fichiers PDF...")
    start_time = time.time()
    
    try:
        documents = SimpleDirectoryReader(
            DATA_PATH,
            recursive=True,
            file_extractor=file_extractor
        ).load_data()
    except Exception as e:
        print(f"❌ Erreur lors de la lecture : {e}")
        return False
    
    read_time = time.time() - start_time
    print(f"📄 {len(documents)} pages chargées en {read_time:.1f}s")
    
    if len(documents) == 0:
        print("⚠️ Aucun document trouvé !")
        return False
    
    # Limiter si demandé
    if max_docs and len(documents) > max_docs:
        print(f"⚠️ Limitation à {max_docs} documents pour ce test")
        documents = documents[:max_docs]
    
    # F. Indexation
    print("\n⚙️ Création des vecteurs (embeddings via API)...")
    print("   ⏱️ Cela peut prendre plusieurs minutes...")
    start_time = time.time()
    
    try:
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
    except Exception as e:
        print(f"❌ Erreur lors de l'indexation : {e}")
        import traceback
        traceback.print_exc()
        return False
    
    index_time = time.time() - start_time
    print(f"\n✅ Indexation terminée en {index_time:.1f}s")
    
    # Vérifier le résultat
    final_count = chroma_collection.count()
    print(f"📊 {final_count} chunks indexés dans ChromaDB")
    print(f"📊 Requêtes API effectuées : {embed_model._request_count}")
    
    print("\n" + "=" * 60)
    print("✅ INGESTION TERMINÉE AVEC SUCCÈS !")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingestion des documents PDF")
    parser.add_argument("--force", action="store_true", help="Force la réindexation")
    parser.add_argument("--max-docs", type=int, help="Limite le nombre de documents")
    args = parser.parse_args()
    
    success = ingest_documents(force=args.force, max_docs=args.max_docs)
    sys.exit(0 if success else 1)
