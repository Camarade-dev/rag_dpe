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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# Import conditionnel des LLMs externes
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # openai, huggingface, anthropic, ollama

if LLM_PROVIDER == "openai":
    from llama_index.llms.openai import OpenAI
elif LLM_PROVIDER == "huggingface":
    try:
        from llama_index.llms.huggingface import HuggingFaceInferenceAPI
    except ImportError:
        from llama_index.llms.huggingface import HuggingFaceLLM
elif LLM_PROVIDER == "anthropic":
    from llama_index.llms.anthropic import Anthropic
elif LLM_PROVIDER == "ollama":
    from llama_index.llms.ollama import Ollama
else:
    # Fallback vers LlamaCPP local si aucun provider externe n'est configuré
    from llama_index.llms.llama_cpp import LlamaCPP

# Chemins
DB_PATH = "./data/chroma_db"
# Chemin du modèle LLM local (utilisé uniquement si LLM_PROVIDER n'est pas configuré)
# Le modèle doit être au format .gguf et placé dans ./data/llm_models/
MODEL_PATH = os.getenv("LLM_MODEL_PATH", "./data/llm_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
COLLECTION_NAME = "renovation_knowledge"
PROMPT_PATH = "./prompts/renovation_expert.txt" # <--- Nouveau chemin vers ton fichier texte

class RenovationRAG:
    def __init__(self):
        print("🔧 Initialisation du moteur RAG...")
        self._init_llm()
        self._init_embedding()
        self._init_vector_store()
        self._init_query_engine()
        print("✅ Moteur RAG prêt à l'emploi !")

    def _init_llm(self):
        """Charge le LLM (externe ou local selon la configuration)"""
        provider = LLM_PROVIDER
        
        if provider == "openai":
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
            
            if api_key:
                print(f"🤖 Utilisation de Hugging Face Inference API : {model_name}")
                # Utiliser l'API Inference de Hugging Face
                try:
                    from llama_index.llms.huggingface import HuggingFaceInferenceAPI
                    self.llm = HuggingFaceInferenceAPI(
                        model_name=model_name,
                        token=api_key,
                        temperature=0.1,
                        max_new_tokens=1024
                    )
                except ImportError:
                    # Fallback si HuggingFaceInferenceAPI n'est pas disponible
                    print("⚠️  HuggingFaceInferenceAPI non disponible, utilisation de HuggingFaceLLM local")
                    self.llm = HuggingFaceLLM(
                        model_name=model_name,
                        temperature=0.1,
                        max_new_tokens=1024,
                        context_window=4096
                    )
            else:
                # Fallback vers modèle local Hugging Face (nécessite plus de RAM)
                print(f"🤖 Utilisation de Hugging Face local : {model_name}")
                print("⚠️  Note: Sans API key, le modèle sera chargé localement (nécessite beaucoup de RAM)")
                self.llm = HuggingFaceLLM(
                    model_name=model_name,
                    temperature=0.1,
                    max_new_tokens=1024,
                    context_window=4096
                )
                
        elif provider == "anthropic":
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
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"❌ Modèle local introuvable : {MODEL_PATH}\n"
                    f"📁 Placez votre fichier .gguf dans : ./data/llm_models/\n"
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
        """Charge le modèle de vectorisation"""
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = self.embed_model

    def _init_vector_store(self):
        """Connexion à ChromaDB"""
        # La télémétrie est désactivée via les variables d'environnement
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