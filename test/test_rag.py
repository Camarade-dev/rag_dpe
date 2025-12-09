from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
import chromadb
import sys

# --- 1. CONFIGURATION DU LLM (MISTRAL) ---
print("⏳ Chargement du LLM...")
llm = LlamaCPP(
    # Assure-toi que le chemin est bon
    model_path="./data/llm_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", 
    temperature=0.1,
    max_new_tokens=1024, # On augmente un peu pour les réponses longues
    context_window=4096,
    model_kwargs={"n_gpu_layers": 0},
    verbose=True # On met False pour avoir moins de blabla technique dans le terminal
)

# --- 2. CONFIGURATION EMBEDDINGS ---
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.llm = llm
Settings.embed_model = embed_model

# --- 3. CONNEXION A LA BASE CHROMA ---
print("⏳ Connexion à la base de données...")
db = chromadb.PersistentClient(path="./data/chroma_db")
chroma_collection = db.get_or_create_collection("renovation_knowledge")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# On enlève persist_dir pour éviter l'erreur docstore.json
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)

# --- 4. DÉFINITION DU PROMPT (POUR FORCER LE FRANÇAIS) ---
# C'est ici qu'on donne l'ordre strict au modèle
template_fr = (
    "Tu es un assistant expert en rénovation énergétique et bâtiment (normes DTU, CPT).\n"
    "Utilise les informations de contexte ci-dessous pour répondre à la question.\n"
    "Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.\n"
    "IMPORTANT : Réponds impérativement en FRANÇAIS.\n"
    "---------------------\n"
    "CONTEXTE :\n"
    "{context_str}\n"
    "---------------------\n"
    "QUESTION : {query_str}\n"
    "RÉPONSE :"
)
qa_template = PromptTemplate(template_fr)

# --- 5. CRÉATION DU MOTEUR AVEC STREAMING ---
query_engine = index.as_query_engine(
    text_qa_template=qa_template, # On applique notre template français
    streaming=True,               # On active le mode "machine à écrire"
    similarity_top_k=3            # On lit les 3 meilleurs passages trouvés
)

# --- 6. TEST ---
question = "Quelles sont les conditions pour l'évacuation des fumées d'une chaudière fioul étanche ?"
print(f"\n❓ Question : {question}\n")
print("💡 Réponse en cours de génération...\n")

# Lancement de la requête
response = query_engine.query(question)

# Affichage en direct (Streaming)
response.print_response_stream()
print("\n") # Petit saut de ligne à la fin