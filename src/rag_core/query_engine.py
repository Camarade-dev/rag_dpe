import sys
import os  # <--- Ajouté pour lire le fichier
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

from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
# Ne pas importer HuggingFaceEmbedding ici (charge torch) - import conditionnel dans _init_embedding
import chromadb

# Import conditionnel des LLMs externes avec gestion d'erreurs robuste
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # openai, huggingface, anthropic, ollama

# Imports conditionnels avec gestion d'erreurs - On essaie d'importer tous les packages disponibles
# pour permettre de changer de provider via les variables d'environnement
OpenAI = None
Anthropic = None
Ollama = None
LlamaCPP = None
HuggingFaceInferenceAPI = None
HuggingFaceLLM = None

# Essayer d'importer tous les packages (certains peuvent ne pas être installés)
try:
    from llama_index.llms.openai import OpenAI
except ImportError:
    pass

try:
    from llama_index.llms.huggingface import HuggingFaceInferenceAPI
except ImportError:
    pass

try:
    from llama_index.llms.huggingface import HuggingFaceLLM
except ImportError:
    pass

try:
    from llama_index.llms.anthropic import Anthropic
except ImportError:
    pass

try:
    from llama_index.llms.ollama import Ollama
except ImportError:
    pass

try:
    from llama_index.llms.llama_cpp import LlamaCPP
except ImportError:
    pass

# Chemins - Utiliser des chemins absolus basés sur le répertoire du script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "data", "chroma_db"))
# Chemin du modèle LLM local (utilisé uniquement si LLM_PROVIDER n'est pas configuré)
MODEL_PATH = os.getenv("LLM_MODEL_PATH", os.path.join(BASE_DIR, "data", "llm_models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"))
COLLECTION_NAME = "renovation_knowledge"
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "renovation_expert.txt")

class RenovationRAG: 
    def __init__(self): 
        print("🔧 Initialisation du moteur RAG...")
        print(f"📊 Variables d'environnement : USE_API_EMBEDDINGS={os.getenv('USE_API_EMBEDDINGS', 'non définie')}")
        print(f"📊 LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'non définie')}")
        
        print("🤖 Étape 1/4 : Initialisation du LLM...")
        self._init_llm()
        
        print("🧠 Étape 2/4 : Initialisation des embeddings...")
        self._init_embedding()
        
        print("💾 Étape 3/4 : Connexion à ChromaDB...")
        self._init_vector_store()
        
        print("🔍 Étape 4/4 : Configuration du query engine...")
        self._init_query_engine()
        print("✅ Moteur RAG prêt à l'emploi !")

    def _init_llm(self):
        """Charge le LLM (externe ou local selon la configuration)"""
        provider = LLM_PROVIDER
         
        if provider == "openai":
            if OpenAI is None:
                raise ImportError("❌ Package llama-index-llms-openai non installé. Installez-le avec: pip install llama-index-llms-openai")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("❌ OPENAI_API_KEY non définie. Configurez-la dans les variables d'environnement.")
            
            model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            print(f"🤖 Utilisation d'OpenAI : {model_name}")
            self.llm = OpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0.1,
                max_tokens=1024
            )
            
        elif provider == "huggingface":
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            model_name = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
            
            if not api_key:
                raise ValueError("❌ HUGGINGFACE_API_KEY non définie. Configurez-la dans les variables d'environnement pour utiliser l'API Inference (gratuit et sans RAM).")
            
            if HuggingFaceInferenceAPI is not None:
                print(f"🤖 Utilisation de Hugging Face Inference API : {model_name}")
                print(f"🔑 API Key détectée : {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
                try:
                    self.llm = HuggingFaceInferenceAPI(
                        model_name=model_name,
                        token=api_key,
                        temperature=0.1,
                        max_new_tokens=1024
                    )
                except Exception as e:
                    raise RuntimeError(f"❌ Erreur lors de l'initialisation de Hugging Face Inference API : {e}\n"
                                     f"💡 Vérifiez que votre clé API est valide et que le modèle {model_name} est accessible.")
            elif HuggingFaceLLM is not None:
                # Fallback vers modèle local Hugging Face (nécessite plus de RAM)
                print(f"⚠️  HuggingFaceInferenceAPI non disponible, utilisation du modèle local : {model_name}")
                print("⚠️  ATTENTION: Le modèle sera chargé localement (nécessite beaucoup de RAM)")
                self.llm = HuggingFaceLLM(
                    model_name=model_name,
                    temperature=0.1,
                    max_new_tokens=1024,
                    context_window=4096
                )
            else:
                raise ImportError("❌ Package llama-index-llms-huggingface non installé.\n"
                                "💡 Installez-le avec: pip install llama-index-llms-huggingface huggingface-hub")
                
        elif provider == "anthropic":
            if Anthropic is None:
                raise ImportError("❌ Package llama-index-llms-anthropic non installé. Installez-le avec: pip install llama-index-llms-anthropic")
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("❌ ANTHROPIC_API_KEY non définie. Configurez-la dans les variables d'environnement.")
            
            model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
            print(f"🤖 Utilisation d'Anthropic Claude : {model_name}")
            self.llm = Anthropic(
                api_key=api_key,
                model=model_name,
                temperature=0.1,
                max_tokens=1024
            )
            
        elif provider == "ollama":
            if Ollama is None:
                raise ImportError("❌ Package llama-index-llms-ollama non installé. Installez-le avec: pip install llama-index-llms-ollama")
            
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model_name = os.getenv("OLLAMA_MODEL", "mistral")
            print(f"🤖 Utilisation d'Ollama : {model_name} ({base_url})")
            self.llm = Ollama(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
                request_timeout=120.0
            )
            
        else:
            # Fallback vers modèle local LlamaCPP
            if LlamaCPP is None:
                raise ImportError("❌ Package llama-index-llms-llama-cpp non installé. Installez-le avec: pip install llama-index-llms-llama-cpp")
            
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"❌ Modèle local introuvable : {MODEL_PATH}\n"
                    f"📁 Placez votre fichier .gguf dans : {os.path.dirname(MODEL_PATH)}\n"
                    f"💡 Ou configurez un LLM externe avec LLM_PROVIDER (openai, huggingface, anthropic, ollama)"
                )
            
            print(f"🤖 Chargement du modèle local : {os.path.basename(MODEL_PATH)}")
            self.llm = LlamaCPP(
                model_path=MODEL_PATH,
                temperature=0.1,
                max_new_tokens=1024,
                context_window=4096,
                model_kwargs={"n_gpu_layers": 0},
                verbose=False
            )
        
        Settings.llm = self.llm
        print("✅ LLM initialisé avec succès")

    def _init_embedding(self):
        """Charge le modèle de vectorisation (API ou local selon configuration)"""
        # Vérifier si on utilise l'API Hugging Face pour les embeddings (évite torch)
        use_api_embeddings = os.getenv("USE_API_EMBEDDINGS", "false").lower() == "true"
        
        if use_api_embeddings:
            # Utiliser l'API Hugging Face (pas de torch nécessaire, économise ~400 MB RAM)
            try:
                # Essayer d'importer HuggingFaceInferenceAPIEmbedding
                from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
                api_key = os.getenv("HUGGINGFACE_API_KEY")
                if not api_key:
                    raise ValueError("❌ HUGGINGFACE_API_KEY requise pour les embeddings API")
                
                self.embed_model = HuggingFaceInferenceAPIEmbedding(
                    api_key=api_key,
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                print("✅ Embeddings via API Hugging Face (pas de modèle en mémoire, économise ~400 MB RAM)")
            except ImportError as e:
                error_msg = f"❌ HuggingFaceInferenceAPIEmbedding non disponible : {e}"
                print(error_msg)
                print("💡 Vérifiez que llama-index-embeddings-huggingface est installé")
                print("💡 Si sentence-transformers n'est pas installé, c'est normal avec requirements_render.txt")
                raise ImportError(f"{error_msg}\n💡 Sur Render, utilisez requirements_render.txt et vérifiez que USE_API_EMBEDDINGS=true")
            except Exception as e:
                raise RuntimeError(f"❌ Erreur lors de l'initialisation des embeddings API : {e}")
        else:
            # Version locale (nécessite sentence-transformers et torch)
            # Import conditionnel pour éviter de charger torch si on ne l'utilise pas
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
                print("📦 Embeddings locaux (modèle chargé en mémoire)")
            except ImportError as e:
                raise ImportError(f"❌ HuggingFaceEmbedding non disponible : {e}\n💡 Installez sentence-transformers avec: pip install sentence-transformers")
        
        Settings.embed_model = self.embed_model

    def _init_vector_store(self):
        """Connexion à ChromaDB"""
        # La télémétrie est désactivée via les variables d'environnement
        # Créer le dossier si nécessaire
        os.makedirs(DB_PATH, exist_ok=True)
        db = chromadb.PersistentClient(path=DB_PATH)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)

    def _init_query_engine(self):
        """Configure le prompt depuis un fichier et le moteur de recherche"""
        index = VectorStoreIndex.from_vector_store(
            self.storage_context.vector_store,
            storage_context=self.storage_context,
        )

        # --- NOUVEAU CODE : Lecture du fichier txt ---
        if not os.path.exists(PROMPT_PATH):
            raise FileNotFoundError(f"❌ Le fichier de prompt est introuvable : {PROMPT_PATH}")

        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            template_content = f.read()
        
        # On vérifie que les variables obligatoires sont bien dans le texte
        if "{context_str}" not in template_content or "{query_str}" not in template_content:
            raise ValueError("❌ Le fichier prompt doit contenir {context_str} et {query_str}")

        qa_template = PromptTemplate(template_content)
        # ---------------------------------------------

        self.query_engine = index.as_query_engine(
            text_qa_template=qa_template,
            streaming=True,
            similarity_top_k=3 # Limite de 3 documents pour plus de contexte
        )

    def query(self, user_question):
        """Méthode publique pour poser une question"""
        return self.query_engine.query(user_question)