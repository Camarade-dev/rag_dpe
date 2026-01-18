"""
API FastAPI pour le système RAG de conseils en rénovation
Accepte une question et retourne une réponse basée sur les documents
"""
import sys
import os
import warnings
# Désactiver les warnings non-critiques
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ALLOW_RESET"] = "TRUE"

# Intercepter les erreurs de télémétrie ChromaDB
import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import sys

# Ajouter le chemin src pour importer le RAG
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag_core.query_engine import RenovationRAG
from pdf_generator import parse_building_info, parse_rag_response, generate_renovation_pdf

# Initialiser le RAG une seule fois au démarrage
rag_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    global rag_engine
    # Startup
    try:
        print("🔧 Initialisation du moteur RAG...")
        rag_engine = RenovationRAG()
        print("✅ Moteur RAG prêt !")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation du RAG: {e}")
        raise
    yield
    # Shutdown (nettoyage si nécessaire)
    pass

app = FastAPI(title="API RAG Rénovation", version="1.0.0", lifespan=lifespan)

# Servir les fichiers PDF statiques
# Calculer le chemin absolu vers le dossier docs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_PATH = os.path.join(BASE_DIR, "docs")
print(f"📂 Chemin docs calculé : {DOCS_PATH}")
print(f"📂 Chemin docs existe : {os.path.exists(DOCS_PATH)}")
 
# Endpoint pour chercher et servir un PDF par son nom (cherche dans tous les sous-dossiers)
@app.get("/docs/{file_name:path}")
async def get_pdf(file_name: str):
    """Cherche et sert un PDF par son nom dans tous les sous-dossiers de docs"""
    import glob
    
    if not os.path.exists(DOCS_PATH):
        raise HTTPException(status_code=500, detail=f"Dossier docs non trouvé : {DOCS_PATH}")
    
    # Nettoyer le nom du fichier (enlever les chemins relatifs malveillants)
    file_name = os.path.basename(file_name)
    
    # Chercher le fichier dans tous les sous-dossiers
    search_pattern = os.path.join(DOCS_PATH, "**", file_name)
    matches = glob.glob(search_pattern, recursive=True)
    
    if matches:
        # Prendre le premier match
        pdf_path = matches[0]
        if os.path.exists(pdf_path):
            print(f"✅ PDF trouvé : {pdf_path}")
            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                filename=os.path.basename(pdf_path),
                headers={"Content-Disposition": f'inline; filename="{os.path.basename(pdf_path)}"'}
            )
    
    print(f"❌ PDF non trouvé : {file_name} dans {DOCS_PATH}")
    raise HTTPException(status_code=404, detail=f"PDF non trouvé : {file_name}")

# CORS pour permettre les requêtes depuis le backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# MODÈLES DE DONNÉES
# ==============================================================================
class RAGRequest(BaseModel):
    """Requête pour le RAG"""
    question: str
    dpe_results: Optional[dict] = None  # Résultats du DPE pour personnaliser la question

class RAGResponse(BaseModel):
    """Réponse du RAG"""
    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None

# ==============================================================================
# ENDPOINTS
# ==============================================================================
@app.get("/")
async def root():
    """Health check"""
    return {"ok": True, "message": "API RAG Rénovation opérationnelle"}

@app.post("/query", response_model=RAGResponse)
async def query_rag(request: RAGRequest):
    """
    Pose une question au système RAG et retourne une réponse avec sources
    """
    global rag_engine
    
    if not rag_engine:
        raise HTTPException(status_code=503, detail="Moteur RAG non initialisé")
    
    try:
        # Construire la question personnalisée si des résultats DPE sont fournis
        question = request.question
        if request.dpe_results:
            # Personnaliser la question avec les résultats du DPE
            classe_dpe = request.dpe_results.get("classe_dpe_finale", "inconnue")
            etiquette_energie = request.dpe_results.get("etiquette_energie", "inconnue")
            
            question = f"""Mon logement a un DPE {classe_dpe} (étiquette énergétique {etiquette_energie}).
{request.question}

Peux-tu me donner des conseils personnalisés de rénovation énergétique adaptés à mon DPE ?"""
        
        # Interroger le RAG
        response = rag_engine.query(question)
        
        # Extraire le texte de la réponse (streaming)
        texte_complet = ""
        if hasattr(response, 'response_gen'):
            for token in response.response_gen:
                texte_complet += token
        else:
            texte_complet = str(response)
        
        # Extraire les sources
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                sources.append({
                    "file_name": node.metadata.get('file_name', 'Inconnu'),
                    "page": node.metadata.get('page_label', '?'),
                    "score": float(node.score) if node.score else 0.0
                })
        
        return RAGResponse(
            ok=True,
            data={
                "response": texte_complet,
                "sources": sources
            }
        )
        
    except Exception as e:
        print(f"❌ Erreur lors de la requête RAG: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la réponse: {str(e)}")

@app.post("/query/pdf")
async def query_rag_and_generate_pdf(request: RAGRequest):
    """
    Pose une question au système RAG, génère une réponse et crée un PDF du rapport
    """
    global rag_engine
    
    if not rag_engine:
        raise HTTPException(status_code=503, detail="Moteur RAG non initialisé")
    
    try:
        # Construire la question personnalisée si des résultats DPE sont fournis
        question = request.question
        if request.dpe_results:
            # Personnaliser la question avec les résultats du DPE
            classe_dpe = request.dpe_results.get("classe_dpe_finale", "inconnue")
            etiquette_energie = request.dpe_results.get("etiquette_energie", "inconnue")
            
            question = f"""Mon logement a un DPE {classe_dpe} (étiquette énergétique {etiquette_energie}).
{request.question}

Peux-tu me donner des conseils personnalisés de rénovation énergétique adaptés à mon DPE ?"""
        
        # Interroger le RAG
        response = rag_engine.query(question)
        
        # Extraire le texte de la réponse (streaming)
        texte_complet = ""
        if hasattr(response, 'response_gen'):
            for token in response.response_gen:
                texte_complet += token
        else:
            texte_complet = str(response)
        
        # Extraire les sources
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                sources.append({
                    "file_name": node.metadata.get('file_name', 'Inconnu'),
                    "page": node.metadata.get('page_label', '?'),
                    "score": float(node.score) if node.score else 0.0
                })
        
        # Générer le PDF
        outputs_dir = os.path.join(BASE_DIR, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        pdf_filename = f"rapport_renovation_{os.urandom(8).hex()}.pdf"
        pdf_path = os.path.join(outputs_dir, pdf_filename)
        
        try:
            # Parser les informations du bâtiment depuis la question
            building_info = parse_building_info(question)
            # Parser la réponse du RAG
            parsed_response = parse_rag_response(texte_complet)
            # Générer le PDF
            generate_renovation_pdf(building_info, parsed_response, pdf_path)
            
            # Retourner le PDF
            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                filename=pdf_filename,
                headers={"Content-Disposition": f'inline; filename="{pdf_filename}"'}
            )
        except Exception as pdf_error:
            print(f"❌ Erreur lors de la génération du PDF: {pdf_error}")
            import traceback
            traceback.print_exc()
            # En cas d'erreur PDF, retourner quand même la réponse texte
            return RAGResponse(
                ok=True,
                data={
                    "response": texte_complet,
                    "sources": sources,
                    "pdf_error": str(pdf_error)
                }
            )
        
    except Exception as e:
        print(f"❌ Erreur lors de la requête RAG: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la réponse: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

