"""
Script d'ingestion robuste compatible avec Render
Utilise les mêmes embeddings API que l'API RAG (pas de torch/sentence-transformers)
Avec gestion des erreurs et retries pour l'API Hugging Face
"""
import os
import sys
import warnings
import time
import random

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


class RobustHuggingFaceEmbedding(BaseEmbedding):
    """
    Wrapper robuste pour l'API Hugging Face embeddings
    Avec retries, backoff exponentiel et gestion des erreurs
    """
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "intfloat/multilingual-e5-base",
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        super().__init__(model_name=model_name)
        object.__setattr__(self, 'api_key', api_key)
        object.__setattr__(self, 'url', f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction")
        object.__setattr__(self, 'headers', {"Authorization": f"Bearer {api_key}"})
        object.__setattr__(self, '_request_count', 0)
        object.__setattr__(self, '_error_count', 0)
        object.__setattr__(self, '_last_request_time', 0)
        object.__setattr__(self, 'max_retries', max_retries)
        object.__setattr__(self, 'base_delay', base_delay)
        object.__setattr__(self, 'max_delay', max_delay)
    
    def _rate_limit(self):
        """Rate limiting conservateur pour éviter les erreurs 429/500"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        # Attendre au minimum 0.5 secondes entre les requêtes (2 req/s max)
        min_delay = 0.5
        if time_since_last < min_delay:
            time.sleep(min_delay - time_since_last)
        
        object.__setattr__(self, '_last_request_time', time.time())
        object.__setattr__(self, '_request_count', self._request_count + 1)
    
    def _get_embedding_with_retry(self, text: str, prefix: str = "passage") -> List[float]:
        """Obtenir l'embedding avec retries et backoff exponentiel"""
        
        # Ajouter le préfixe pour multilingual-e5-base
        if not text.strip().startswith("query:") and not text.strip().startswith("passage:"):
            text = f"{prefix}: {text}"
        
        payload = {"inputs": text}
        last_error = None
        
        for attempt in range(self.max_retries):
            self._rate_limit()
            
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                # Gérer les différents codes d'erreur
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, list):
                        embedding = data[0] if isinstance(data[0], list) else data
                        return [float(x) for x in embedding]
                    raise ValueError(f"Format de réponse inattendu: {type(data)}")
                
                elif response.status_code == 503:
                    # Modèle en cours de chargement
                    wait_time = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    print(f"   ⏳ Modèle en chargement (503), attente {wait_time:.1f}s... (tentative {attempt+1}/{self.max_retries})")
                    time.sleep(wait_time)
                    last_error = f"503 Service Unavailable"
                    continue
                
                elif response.status_code == 500:
                    # Erreur serveur interne - retry avec backoff
                    wait_time = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    print(f"   ⚠️ Erreur serveur (500), retry dans {wait_time:.1f}s... (tentative {attempt+1}/{self.max_retries})")
                    object.__setattr__(self, '_error_count', self._error_count + 1)
                    time.sleep(wait_time)
                    last_error = f"500 Internal Server Error"
                    continue
                
                elif response.status_code == 429:
                    # Rate limiting - attendre plus longtemps
                    wait_time = min(self.base_delay * (3 ** attempt) + random.uniform(0, 2), self.max_delay)
                    print(f"   🚫 Rate limit (429), attente {wait_time:.1f}s... (tentative {attempt+1}/{self.max_retries})")
                    time.sleep(wait_time)
                    last_error = f"429 Too Many Requests"
                    continue
                
                else:
                    # Autre erreur
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                wait_time = min(self.base_delay * (2 ** attempt), self.max_delay)
                print(f"   ⏱️ Timeout, retry dans {wait_time:.1f}s... (tentative {attempt+1}/{self.max_retries})")
                time.sleep(wait_time)
                last_error = "Timeout"
                continue
                
            except requests.exceptions.RequestException as e:
                wait_time = min(self.base_delay * (2 ** attempt), self.max_delay)
                print(f"   ❌ Erreur réseau: {e}, retry dans {wait_time:.1f}s... (tentative {attempt+1}/{self.max_retries})")
                time.sleep(wait_time)
                last_error = str(e)
                continue
        
        # Toutes les tentatives ont échoué
        raise RuntimeError(f"Échec après {self.max_retries} tentatives. Dernière erreur: {last_error}")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding_with_retry(query, prefix="query")
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding_with_retry(text, prefix="passage")
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._aget_query_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Obtenir les embeddings de plusieurs textes avec progression"""
        embeddings = []
        total = len(texts)
        start_time = time.time()
        
        for i, text in enumerate(texts):
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                print(f"   📊 Embedding {i}/{total} ({i*100//total}%) - {rate:.1f}/s - ~{remaining:.0f}s restantes")
            
            try:
                embedding = self._get_text_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"   ❌ Erreur définitive sur le texte {i}: {e}")
                # Utiliser un embedding vide ou lever l'exception
                raise
        
        return embeddings
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)
    
    def get_stats(self):
        """Retourne les statistiques d'utilisation"""
        return {
            "requests": self._request_count,
            "errors": self._error_count
        }


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
    Indexe les documents PDF dans ChromaDB avec gestion robuste des erreurs
    
    Args:
        force: Si True, réindexe même si des documents existent déjà
        max_docs: Limite le nombre de documents à indexer
    """
    print("=" * 60)
    print("🚀 INGESTION ROBUSTE DES DOCUMENTS")
    print("=" * 60)
    
    # Vérifier les variables d'environnement
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("❌ HUGGINGFACE_API_KEY non définie !")
        return False
    
    print(f"📂 Dossier documents : {DATA_PATH}")
    print(f"💾 Base ChromaDB : {DB_PATH}")
    print(f"🔑 API Key : {api_key[:10]}...{api_key[-4:]}")
    
    # Vérifier si l'ingestion est nécessaire
    if not force and not check_collection_empty(DB_PATH, COLLECTION_NAME):
        print("✅ La collection contient déjà des documents.")
        return True
    
    # Vérifier que le dossier docs existe
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dossier documents introuvable : {DATA_PATH}")
        return False
    
    # A. Modèle d'Embedding robuste via API
    print("\n🧠 Initialisation des embeddings robustes...")
    try:
        embed_model = RobustHuggingFaceEmbedding(
            api_key=api_key,
            model_name="intfloat/multilingual-e5-base",
            max_retries=5,
            base_delay=2.0,
            max_delay=60.0
        )
        Settings.embed_model = embed_model
        print("✅ Embeddings initialisés avec retries")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation des embeddings : {e}")
        return False
    
    # B. Chunking - Réduire la taille pour moins de tokens
    print("\n📏 Configuration du chunking...")
    text_splitter = SentenceSplitter(
        chunk_size=256,  # Réduit de 512 à 256 pour des chunks plus petits
        chunk_overlap=25
    )
    Settings.text_splitter = text_splitter
    print("✅ Chunking configuré (256 tokens, overlap 25)")
    
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
        print(f"⚠️ Limitation à {max_docs} documents")
        documents = documents[:max_docs]
    
    # F. Indexation par petits batches
    print("\n⚙️ Indexation par batches...")
    batch_size = 10  # Petits batches pour éviter les timeouts
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    print(f"   📦 {len(documents)} documents en {total_batches} batches de {batch_size}")
    
    successful_docs = 0
    failed_batches = 0
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(documents))
        batch_docs = documents[start_idx:end_idx]
        
        print(f"\n📦 Batch {batch_idx + 1}/{total_batches} (docs {start_idx + 1}-{end_idx})...")
        
        try:
            # Indexer ce batch
            VectorStoreIndex.from_documents(
                batch_docs,
                storage_context=storage_context,
                show_progress=True
            )
            successful_docs += len(batch_docs)
            print(f"   ✅ Batch {batch_idx + 1} terminé ({successful_docs} docs indexés)")
            
            # Pause entre les batches pour éviter le rate limiting
            if batch_idx < total_batches - 1:
                pause = 2.0  # 2 secondes entre les batches
                print(f"   ⏸️ Pause de {pause}s avant le prochain batch...")
                time.sleep(pause)
                
        except Exception as e:
            failed_batches += 1
            print(f"   ❌ Erreur batch {batch_idx + 1}: {e}")
            
            # Continuer avec le batch suivant après une pause plus longue
            if failed_batches < 3:  # Tolérer jusqu'à 3 erreurs
                print(f"   🔄 Continuation après erreur ({failed_batches}/3 tolérées)...")
                time.sleep(10)  # Pause plus longue après une erreur
                continue
            else:
                print(f"   🛑 Trop d'erreurs, arrêt de l'ingestion")
                break
    
    # Résumé
    stats = embed_model.get_stats()
    final_count = chroma_collection.count()
    
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE L'INGESTION")
    print("=" * 60)
    print(f"   📄 Documents traités: {successful_docs}/{len(documents)}")
    print(f"   📦 Batches échoués: {failed_batches}")
    print(f"   🔢 Chunks dans ChromaDB: {final_count}")
    print(f"   🌐 Requêtes API: {stats['requests']}")
    print(f"   ⚠️ Erreurs API (récupérées): {stats['errors']}")
    
    if final_count > 0:
        print("\n✅ INGESTION TERMINÉE AVEC SUCCÈS !")
        return True
    else:
        print("\n❌ INGESTION ÉCHOUÉE - Aucun chunk indexé")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingestion robuste des documents PDF")
    parser.add_argument("--force", action="store_true", help="Force la réindexation")
    parser.add_argument("--max-docs", type=int, default=30, help="Limite le nombre de documents (défaut: 30)")
    args = parser.parse_args()
    
    success = ingest_documents(force=args.force, max_docs=args.max_docs)
    sys.exit(0 if success else 1)
