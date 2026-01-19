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
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
# Ne pas importer HuggingFaceEmbedding ici (charge torch) - import conditionnel dans _init_embedding
import chromadb
import asyncio
from typing import List, Optional

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

# Essayer d'abord la nouvelle API recommandée
try:
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    _USE_NEW_HF_LLM_API = True
except ImportError:
    # Fallback vers l'ancienne API (dépréciée mais fonctionnelle)
    try:
        from llama_index.llms.huggingface import HuggingFaceInferenceAPI
        _USE_NEW_HF_LLM_API = False
    except ImportError:
        HuggingFaceInferenceAPI = None
        _USE_NEW_HF_LLM_API = None

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


class HuggingFaceTextEmbeddingsWrapper(BaseEmbedding):
    """
    Wrapper personnalisé pour utiliser l'API Hugging Face embeddings
    via le nouveau router.huggingface.co (obligatoire depuis 2025)
    """
    def __init__(self, api_key: str, model_name: str = "intfloat/multilingual-e5-base"):
        try:
            import requests
            self.requests = requests
            self.api_key = api_key
            self.model_name = model_name
            # Utiliser UNIQUEMENT le nouveau router (l'ancienne API api-inference.huggingface.co est dépréciée depuis 2025)
            # Format correct: https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction
            self.url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
            self.model_name = model_name
            self.headers = {
                "Authorization": f"Bearer {api_key}"
            }
            # Appeler le constructeur parent
            super().__init__(model_name=model_name)
        except ImportError:
            raise ImportError("❌ requests n'est pas installé. Installez-le avec: pip install requests")
        except Exception as e:
            raise RuntimeError(f"❌ Erreur lors de l'initialisation du wrapper : {e}")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Obtenir l'embedding d'une requête (synchrone) via router.huggingface.co"""
        # IMPORTANT: Pour multilingual-e5-base, ajouter le préfixe "query: " ou "passage: "
        # Voir: https://huggingface.co/intfloat/multilingual-e5-base
        # Pour les requêtes de recherche, on utilise "query: "
        if not query.strip().startswith("query:") and not query.strip().startswith("passage:"):
            query = f"query: {query}"
        
        # Format correct pour router.huggingface.co/hf-inference/models/.../pipeline/feature-extraction
        payload = {
            "inputs": query  # Texte simple (pas une liste)
        }
        
        try:
            response = self.requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Gérer les erreurs spécifiques
            if response.status_code == 410:
                raise RuntimeError(
                    f"❌ L'endpoint {self.url} est déprécié (410 Gone). "
                    f"Utilisez router.huggingface.co (déjà configuré). "
                    f"Vérifiez que votre token API a les permissions nécessaires."
                )
            
            response.raise_for_status()
            data = response.json()
            
            # Parser la réponse (format router.huggingface.co: [[...]] ou [...])
            if isinstance(data, list):
                # Format standard: liste de listes ou liste directe
                if len(data) > 0:
                    # Si c'est une liste de listes, prendre le premier élément
                    embedding = data[0] if isinstance(data[0], list) else data
                    return [float(x) for x in embedding]
                raise ValueError("Réponse vide")
            elif isinstance(data, dict):
                # Chercher dans différentes clés possibles (fallback)
                for key in ["embeddings", "data", "embedding", "vector"]:
                    if key in data:
                        emb = data[key]
                        if isinstance(emb, list) and len(emb) > 0:
                            result = emb[0] if isinstance(emb[0], list) else emb
                            return [float(x) for x in result]
                raise ValueError(f"Format de réponse inattendu: {list(data.keys())}")
            else:
                # Réponse directe (array numpy ou similaire)
                return [float(x) for x in data] if hasattr(data, '__iter__') else [float(data)]
                
        except self.requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                try:
                    error_detail = e.response.json()
                    error_msg = f"{error_msg} - {error_detail}"
                except:
                    error_msg = f"{error_msg} - Status: {status_code}"
            
            raise RuntimeError(
                f"❌ Impossible d'obtenir l'embedding depuis {self.url}. "
                f"Erreur: {error_msg}. "
                f"Modèle: {self.model_name}. "
                f"Vérifiez que votre token API est valide et a les permissions nécessaires."
            )
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Obtenir l'embedding d'une requête (asynchrone)"""
        # Exécuter la méthode synchrone dans un thread pour éviter les conflits d'event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Obtenir l'embedding d'un texte (même logique que query)"""
        return self._get_query_embedding(text)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Obtenir l'embedding d'un texte (asynchrone)"""
        return await self._aget_query_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Obtenir les embeddings de plusieurs textes (batch)"""
        embeddings = []
        for text in texts:
            embeddings.append(self._get_text_embedding(text))
        return embeddings
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Obtenir les embeddings de plusieurs textes (batch, asynchrone)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)

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
            model_name = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
            
            if not api_key:
                raise ValueError("❌ HUGGINGFACE_API_KEY non définie. Configurez-la dans les variables d'environnement pour utiliser l'API Inference (gratuit et sans RAM).")
            
            if HuggingFaceInferenceAPI is not None:
                if _USE_NEW_HF_LLM_API:
                    print("📦 Utilisation de llama-index-llms-huggingface-api (nouvelle API)")
                else:
                    print("⚠️  Utilisation de llama-index-llms-huggingface (ancienne API, dépréciée)")
                print(f"🤖 Utilisation de Hugging Face Inference API : {model_name}")
                print(f"🔑 API Key détectée : {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
                try:
                    # Les nouvelles classes utilisent 'model_name' et 'token' (ou 'api_key' selon la version)
                    # Essayons d'abord avec 'model_name' et 'token' (nouvelle API)
                    try:
                        self.llm = HuggingFaceInferenceAPI(
                            model_name=model_name,
                            token=api_key,
                            temperature=0.1,
                            max_new_tokens=256  # Réduit pour accélérer et économiser les tokens API
                        )
                    except TypeError:
                        # Si ça ne marche pas, essayons avec 'api_key' (ancienne API)
                        self.llm = HuggingFaceInferenceAPI(
                            model_name=model_name,
                            api_key=api_key,
                            temperature=0.1,
                            max_new_tokens=256
                        )
                except Exception as e:
                    raise RuntimeError(f"❌ Erreur lors de l'initialisation de Hugging Face Inference API : {e}\n"
                                     f"💡 Vérifiez que votre clé API est valide et que le modèle {model_name} est accessible.\n"
                                     f"💡 Assurez-vous que llama-index-llms-huggingface-api est installé.")
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
                raise ImportError("❌ Package llama-index-llms-huggingface-api non installé.\n"
                                "💡 Installez-le avec: pip install llama-index-llms-huggingface-api huggingface-hub")
                
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
                # Essayer d'importer HuggingFaceInferenceAPIEmbedding (nouvelle API)
                # D'abord essayer la nouvelle API recommandée
                HuggingFaceInferenceAPIEmbedding = None
                use_new_embedding_api = False
                try:
                    from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
                    use_new_embedding_api = True
                except ImportError:
                    # Fallback vers l'ancienne API (dépréciée mais fonctionnelle)
                    try:
                        from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
                        use_new_embedding_api = False
                    except ImportError:
                        HuggingFaceInferenceAPIEmbedding = None
                
                api_key = os.getenv("HUGGINGFACE_API_KEY")
                if not api_key:
                    raise ValueError("❌ HUGGINGFACE_API_KEY requise pour les embeddings API")
                
                # Utiliser un modèle compatible avec l'API Hugging Face Inference
                embedding_model_name = os.getenv(
                    "HUGGINGFACE_EMBEDDING_MODEL", 
                    "intfloat/multilingual-e5-base"  # Modèle qui fonctionne avec l'API text-embeddings
                )
                print(f"📦 Modèle d'embedding: {embedding_model_name}")
                
                # PRIORITÉ 1: Utiliser notre wrapper personnalisé (corrige l'erreur 410 Gone)
                print("📦 Utilisation du wrapper personnalisé HuggingFaceTextEmbeddingsWrapper (nouvelle API router.huggingface.co)")
                try:
                    self.embed_model = HuggingFaceTextEmbeddingsWrapper(
                        api_key=api_key,
                        model_name=embedding_model_name
                    )
                    print("✅ Embeddings via API Hugging Face (wrapper personnalisé, pas de modèle en mémoire, économise ~400 MB RAM)")
                except Exception as e:
                    print(f"⚠️  Le wrapper personnalisé a échoué: {e}")
                    print("🔄 Tentative avec les classes llama-index...")
                    
                    # PRIORITÉ 2: Essayer les classes llama-index si disponibles
                    if HuggingFaceInferenceAPIEmbedding is not None:
                        if use_new_embedding_api:
                            print("📦 Utilisation de llama-index-embeddings-huggingface-api (nouvelle API)")
                        else:
                            print("⚠️  Utilisation de llama-index-embeddings-huggingface (ancienne API, dépréciée)")
                        
                        try:
                            # Les nouvelles classes peuvent utiliser 'api_key' ou 'token'
                            try:
                                self.embed_model = HuggingFaceInferenceAPIEmbedding(
                                    api_key=api_key,
                                    model_name=embedding_model_name
                                )
                            except TypeError:
                                # Si ça ne marche pas, essayons avec 'token'
                                self.embed_model = HuggingFaceInferenceAPIEmbedding(
                                    token=api_key,
                                    model_name=embedding_model_name
                                )
                            print("✅ Embeddings via API Hugging Face (llama-index, pas de modèle en mémoire, économise ~400 MB RAM)")
                        except Exception as e2:
                            # Si le modèle ne fonctionne pas, essayons un autre
                            print(f"⚠️  Modèle {embedding_model_name} ne fonctionne pas: {e2}")
                            print("🔄 Tentative avec BAAI/bge-small-en-v1.5...")
                            try:
                                try:
                                    self.embed_model = HuggingFaceInferenceAPIEmbedding(
                                        api_key=api_key,
                                        model_name="BAAI/bge-small-en-v1.5"
                                    )
                                except TypeError:
                                    self.embed_model = HuggingFaceInferenceAPIEmbedding(
                                        token=api_key,
                                        model_name="BAAI/bge-small-en-v1.5"
                                    )
                                print("✅ Modèle BAAI/bge-small-en-v1.5 sélectionné")
                            except Exception as e3:
                                print(f"❌ BAAI/bge-small-en-v1.5 ne fonctionne pas non plus: {e3}")
                                raise RuntimeError(f"❌ Impossible de trouver un modèle d'embedding compatible. Erreurs: {e}, {e2}, {e3}")
                    else:
                        # Si aucune classe llama-index n'est disponible, on a déjà le wrapper personnalisé qui devrait fonctionner
                        raise RuntimeError(f"❌ Aucune méthode d'embedding API n'est disponible. Erreur wrapper: {e}")
            except ImportError as e:
                error_msg = f"❌ HuggingFaceInferenceAPIEmbedding non disponible : {e}"
                print(error_msg)
                print("💡 Vérifiez que llama-index-embeddings-huggingface-api est installé")
                print("💡 Sur Render, utilisez requirements_render.txt et vérifiez que USE_API_EMBEDDINGS=true")
                raise ImportError(f"{error_msg}\n💡 Installez avec: pip install llama-index-embeddings-huggingface-api")
            except Exception as e:
                raise RuntimeError(f"❌ Erreur lors de l'initialisation des embeddings API : {e}")
        else:
            # Version locale (nécessite sentence-transformers et torch)
            # Import conditionnel pour éviter de charger torch si on ne l'utilise pas
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                # Utiliser le même modèle pour la cohérence si on utilise les embeddings locaux
                self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                print("📦 Embeddings locaux (modèle chargé en mémoire)")
            except ImportError as e:
                raise ImportError(f"❌ HuggingFaceEmbedding non disponible : {e}\n💡 Installez sentence-transformers avec: pip install sentence-transformers")
        
        Settings.embed_model = self.embed_model

    def _init_vector_store(self):
        """Connexion à ChromaDB"""
        try:
            print(f"💾 Connexion à ChromaDB dans : {DB_PATH}")
            # La télémétrie est désactivée via les variables d'environnement
            # Créer le dossier si nécessaire
            os.makedirs(DB_PATH, exist_ok=True)
            print("📂 Dossier ChromaDB créé/vérifié")
            
            db = chromadb.PersistentClient(path=DB_PATH)
            print("✅ Client ChromaDB créé")
            
            # Créer la collection avec une embedding function vide
            # On utilise les embeddings de llama-index, pas ceux de ChromaDB
            # Mais ChromaDB nécessite une embedding function pour initialiser la collection
            try:
                # Essayer avec embedding_function=None (certaines versions le supportent)
                chroma_collection = db.get_or_create_collection(
                    COLLECTION_NAME,
                    embedding_function=None
                )
            except (TypeError, ValueError):
                # Si None n'est pas accepté, utiliser une embedding function vide
                # qui ne sera jamais appelée car llama-index gère les embeddings
                class EmptyEmbeddingFunction:
                    def __call__(self, input):
                        # Ne devrait jamais être appelée car llama-index gère les embeddings
                        raise NotImplementedError("Cette embedding function ne doit pas être utilisée")
                
                try:
                    chroma_collection = db.get_or_create_collection(
                        COLLECTION_NAME,
                        embedding_function=EmptyEmbeddingFunction()
                    )
                except Exception:
                    # Dernier recours : créer sans embedding function explicite
                    # Cela utilisera DefaultEmbeddingFunction mais on a installé onnxruntime
                    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
            print(f"✅ Collection '{COLLECTION_NAME}' créée/récupérée")
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            print("✅ VectorStore créé")
            
            self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
            print("✅ StorageContext créé")
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation de ChromaDB : {e}")
            import traceback
            traceback.print_exc()
            raise

    def _init_query_engine(self):
        """Configure le prompt depuis un fichier et le moteur de recherche"""
        try:
            print(f"🔍 Création de l'index depuis le vector store...")
            index = VectorStoreIndex.from_vector_store(
                self.storage_context.vector_store,
                storage_context=self.storage_context,
            )
            print("✅ Index créé")

            # --- NOUVEAU CODE : Lecture du fichier txt ---
            print(f"📄 Lecture du prompt depuis : {PROMPT_PATH}")
            if not os.path.exists(PROMPT_PATH):
                raise FileNotFoundError(f"❌ Le fichier de prompt est introuvable : {PROMPT_PATH}")

            with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                template_content = f.read()
            print("✅ Prompt lu")
            
            # On vérifie que les variables obligatoires sont bien dans le texte
            if "{context_str}" not in template_content or "{query_str}" not in template_content:
                raise ValueError("❌ Le fichier prompt doit contenir {context_str} et {query_str}")

            qa_template = PromptTemplate(template_content)
            print("✅ Template créé")
            # ---------------------------------------------

            print("🔧 Configuration du query engine...")
            self.query_engine = index.as_query_engine(
                text_qa_template=qa_template,
                streaming=True,
                similarity_top_k=2  # Réduit à 2 pour accélérer (était 3)
            )
            print("✅ Query engine configuré")
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation du query engine : {e}")
            import traceback
            traceback.print_exc()
            raise

    def query(self, user_question):
        """Méthode publique pour poser une question"""
        import threading
        import asyncio
        
        # Créer un nouvel event loop dans un thread séparé pour éviter le conflit
        # avec l'event loop de FastAPI/uvicorn
        result_container = {}
        exception_container = {}
        
        def run_in_new_loop():
            """Exécute la requête dans un nouvel event loop isolé"""
            try:
                # Créer un nouvel event loop pour ce thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    # Exécuter la requête (qui peut utiliser asyncio.run() en interne)
                    result = self.query_engine.query(user_question)
                    result_container['result'] = result
                finally:
                    new_loop.close()
            except Exception as e:
                exception_container['exception'] = e
        
        # Exécuter dans un thread séparé avec un nouvel event loop
        thread = threading.Thread(target=run_in_new_loop, daemon=True)
        thread.start()
        thread.join(timeout=600)  # Timeout de 10 minutes
        
        if thread.is_alive():
            raise TimeoutError("La requête RAG a pris plus de 10 minutes")
        
        if 'exception' in exception_container:
            raise exception_container['exception']
        
        if 'result' in result_container:
            return result_container['result']
        
        raise RuntimeError("La requête RAG n'a retourné aucun résultat")