import sys
import os
import warnings
import logging
import threading
import asyncio

# Désactivation des warnings et télémétrie
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Imports spécifiques API (Légers - Pas de torch)
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "data", "chroma_db"))
COLLECTION_NAME = "renovation_knowledge"
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "renovation_expert.txt")

class RenovationRAG: 
    def __init__(self): 
        print("============================================================")
        print("🔧 INITIALISATION DU MOTEUR RAG (MODE API)")
        print("============================================================")
        
        self._init_llm()
        self._init_embedding()
        self._init_vector_store()
        self._init_query_engine()
        
        print("✅ Moteur RAG prêt (Consommation RAM optimisée)")
        print("============================================================")

    def _init_llm(self):
        """Initialise le LLM via Hugging Face Inference API"""
        print("🤖 Étape 1/4 : Initialisation du LLM (API)...")
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        model_name = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        
        if not api_key:
            raise ValueError("❌ HUGGINGFACE_API_KEY manquante dans les variables d'environnement")

        self.llm = HuggingFaceInferenceAPI(
            model_name=model_name,
            token=api_key,
            temperature=0.1,
            max_new_tokens=512
        )
        Settings.llm = self.llm
        print(f"✅ LLM configuré : {model_name}")

    def _init_embedding(self):
        """Initialise les Embeddings via API (CORRECTION ERREUR 410)"""
        print("🧠 Étape 2/4 : Initialisation des embeddings (API)...")
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        # Le modèle e5 est excellent pour le français
        model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        
        # FORÇAGE DE L'URL POUR ÉVITER LE 410 GONE
        # On utilise /models/ au lieu de /pipeline/
        forced_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        self.embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name=model_name,
            token=api_key,
            base_url=forced_url
        )
        Settings.embed_model = self.embed_model
        print(f"✅ Embeddings configurés sur : {forced_url}")

    def _init_vector_store(self):
        """Connexion à ChromaDB"""
        print("💾 Étape 3/4 : Connexion à ChromaDB...")
        os.makedirs(DB_PATH, exist_ok=True)
        
        db = chromadb.PersistentClient(path=DB_PATH)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print(f"✅ ChromaDB prêt (Dossier: {DB_PATH})")

    def _init_query_engine(self):
        """Configuration finale du moteur de recherche"""
        print("🔍 Étape 4/4 : Configuration du query engine...")
        index = VectorStoreIndex.from_vector_store(
            self.storage_context.vector_store,
            storage_context=self.storage_context,
        )

        if os.path.exists(PROMPT_PATH):
            with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                template_content = f.read()
            qa_template = PromptTemplate(template_content)
            print("📄 Prompt personnalisé chargé")
        else:
            print("⚠️ Prompt par défaut utilisé (fichier non trouvé)")
            qa_template = None

        self.query_engine = index.as_query_engine(
            text_qa_template=qa_template,
            streaming=True,
            similarity_top_k=2
        )
        print("✅ Moteur configuré avec succès")

    def query(self, user_question):
        """Méthode de requête avec gestion d'event loop pour FastAPI"""
        result_container = {}
        exception_container = {}
        
        def run_in_new_loop():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    # Appel synchrone de LlamaIndex dans cet loop
                    result = self.query_engine.query(user_question)
                    result_container['result'] = result
                finally:
                    new_loop.close()
            except Exception as e:
                exception_container['exception'] = e
        
        thread = threading.Thread(target=run_in_new_loop, daemon=True)
        thread.start()
        thread.join(timeout=300) 
        
        if 'exception' in exception_container:
            raise exception_container['exception']
        return result_container.get('result')