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
# Le chemin parent de api/ est src/, donc on ajoute src/ au path
src_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, src_path)
from rag_core.query_engine import RenovationRAG
from pdf_generator import parse_building_info, parse_rag_response, generate_renovation_pdf

# Initialiser le RAG une seule fois au démarrage
rag_engine = None

def check_and_ingest_if_needed():
    """
    Vérifie si la collection ChromaDB est vide et lance l'ingestion si nécessaire.
    Retourne True si l'ingestion a été effectuée ou si des documents existent déjà.
    """
    import chromadb
    
    db_path = os.getenv("CHROMA_DB_PATH", "/tmp/chroma_db")
    collection_name = "renovation_knowledge"
    
    print(f"🔍 Vérification de la base ChromaDB : {db_path}")
    
    try:
        # Créer le dossier si nécessaire
        os.makedirs(db_path, exist_ok=True)
        
        # Vérifier si la collection existe et contient des documents
        db = chromadb.PersistentClient(path=db_path)
        try:
            collection = db.get_collection(collection_name)
            count = collection.count()
            print(f"📊 Collection '{collection_name}' contient {count} documents")
            
            if count > 0:
                print("✅ Des documents existent déjà, pas besoin d'ingestion")
                return True
        except Exception as e:
            print(f"⚠️ Collection non trouvée ou erreur : {e}")
            count = 0
        
        # Si la collection est vide, lancer l'ingestion
        print("\n" + "=" * 60)
        print("⚠️ COLLECTION VIDE - LANCEMENT DE L'INGESTION AUTOMATIQUE")
        print("=" * 60)
        
        # Vérifier si le dossier docs existe
        docs_path = os.path.join(BASE_DIR, "docs")
        if not os.path.exists(docs_path):
            print(f"❌ Dossier documents introuvable : {docs_path}")
            print("💡 L'API fonctionnera mais sans documents de contexte")
            return False
        
        # Importer et lancer l'ingestion
        try:
            # Ajouter le chemin pour l'import
            ingestion_path = os.path.join(os.path.dirname(__file__), "..", "ingestion")
            sys.path.insert(0, ingestion_path)
            
            from ingest_api import ingest_documents
            
            # Lancer l'ingestion avec un nombre limité de documents pour le premier démarrage
            # Réduit à 30 par défaut pour éviter les timeouts et erreurs API
            max_docs = int(os.getenv("INGESTION_MAX_DOCS", "30"))
            print(f"📄 Ingestion limitée à {max_docs} documents maximum")
            print(f"⚠️ L'ingestion utilise des retries robustes (peut prendre 5-10 min)")
            
            success = ingest_documents(force=False, max_docs=max_docs)
            
            if success:
                print("✅ Ingestion automatique terminée avec succès")
                return True
            else:
                print("❌ Échec de l'ingestion automatique")
                return False
                
        except ImportError as e:
            print(f"❌ Module d'ingestion non trouvé : {e}")
            return False
        except Exception as e:
            print(f"❌ Erreur lors de l'ingestion : {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de la vérification ChromaDB : {e}")
        return False


async def _initialize_rag_background():
    """Initialise le RAG en arrière-plan (non-bloquant)"""
    global rag_engine
    try:
        print("=" * 60)
        print("🔧 Initialisation du moteur RAG en arrière-plan...")
        print("=" * 60)
        print(f"📊 USE_API_EMBEDDINGS={os.getenv('USE_API_EMBEDDINGS', 'non définie')}")
        print(f"📊 LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'non définie')}")
        print(f"📊 HUGGINGFACE_API_KEY={'✅ configurée' if os.getenv('HUGGINGFACE_API_KEY') else '❌ non configurée'}")
        
        # Vérifier si torch est installé (ne devrait pas l'être avec USE_API_EMBEDDINGS=true)
        try:
            import torch
            print(f"⚠️  AVERTISSEMENT: torch est installé (version {torch.__version__}) - cela utilise ~400 MB RAM")
        except ImportError:
            print("✅ torch n'est pas installé - bonne configuration pour économiser la RAM")
        
        # NOUVEAU: Vérifier et indexer les documents si nécessaire
        auto_ingest = os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower() == "true"
        if auto_ingest:
            print("\n🔄 Vérification de l'ingestion des documents...")
            check_and_ingest_if_needed()
        else:
            print("\n⚠️ AUTO_INGEST_ON_STARTUP=false - pas d'ingestion automatique")
        
        print("\n🚀 Démarrage de l'initialisation du RAG...")
        rag_engine = RenovationRAG()
        print("=" * 60)
        print("✅ Moteur RAG prêt !")
        print("=" * 60)
        port = os.getenv("PORT", "8002")
        print(f"🌐 L'API est prête à recevoir des requêtes sur le port {port}")
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ ERREUR CRITIQUE lors de l'initialisation du RAG:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        print("=" * 60)
        # Ne pas raise pour permettre à l'API de démarrer quand même
        # Les requêtes retourneront une erreur 503 mais l'API sera accessible
        print("⚠️  L'API démarre mais le RAG n'est pas initialisé - les requêtes échoueront")
        rag_engine = None
        port = os.getenv("PORT", "8002")
        print(f"🌐 L'API est quand même accessible sur le port {port} (mais le RAG ne fonctionnera pas)")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application - démarrage rapide pour Render"""
    global rag_engine
    rag_engine = None  # Initialiser à None
    
    # Démarrer l'initialisation du RAG en arrière-plan (non-bloquant)
    # Cela permet à Render de détecter le port rapidement
    import asyncio
    asyncio.create_task(_initialize_rag_background())
    
    port = os.getenv("PORT", "8002")
    print(f"🚀 API démarrée rapidement sur le port {port}")
    print(f"⏳ Initialisation du RAG en cours en arrière-plan...")
    print(f"💡 L'endpoint /health répond immédiatement")
    
    yield
    # Shutdown (nettoyage si nécessaire)
    pass

app = FastAPI(title="API RAG Rénovation", version="1.0.0", lifespan=lifespan)

# Endpoint de health check pour Render (répond immédiatement même pendant l'initialisation)
@app.get("/health")
async def health_check():
    """Endpoint de health check pour Render - répond immédiatement"""
    return {
        "status": "ok",
        "rag_initialized": rag_engine is not None,
        "port": os.getenv("PORT", "8002"),
        "message": "API is running" if rag_engine is None else "API and RAG are ready"
    }

# Endpoint racine pour vérifier que l'API répond
@app.get("/")
async def root():
    """Endpoint racine - répond immédiatement"""
    return {
        "status": "ok",
        "service": "RAG API",
        "rag_ready": rag_engine is not None
    }

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
    from urllib.parse import unquote
    
    if not os.path.exists(DOCS_PATH):
        raise HTTPException(status_code=500, detail=f"Dossier docs non trouvé : {DOCS_PATH}")
    
    # Décoder l'URL (gère les %20, %2E, etc.)
    file_name = unquote(file_name)
    
    # Nettoyer le nom du fichier (enlever les chemins relatifs malveillants)
    file_name = os.path.basename(file_name)
    
    print(f"🔍 Recherche du PDF : {file_name}")
    
    # Chercher le fichier dans tous les sous-dossiers
    search_pattern = os.path.join(DOCS_PATH, "**", file_name)
    matches = glob.glob(search_pattern, recursive=True)
    
    # Si pas de match exact, essayer une recherche insensible à la casse
    if not matches:
        print(f"⚠️ Pas de match exact, recherche insensible à la casse...")
        for root, dirs, files in os.walk(DOCS_PATH):
            for file in files:
                if file.lower() == file_name.lower():
                    matches.append(os.path.join(root, file))
                    break
    
    # Si toujours pas de match, essayer de chercher avec des variations d'espaces
    if not matches:
        print(f"⚠️ Pas de match, recherche avec variations d'espaces...")
        # Remplacer les espaces par des underscores et vice versa
        variations = [
            file_name.replace(' ', '_'),
            file_name.replace('_', ' '),
            file_name.replace('%20', ' '),
            file_name.replace(' ', '%20'),
        ]
        for variation in variations:
            if variation != file_name:
                search_pattern = os.path.join(DOCS_PATH, "**", variation)
                matches = glob.glob(search_pattern, recursive=True)
                if matches:
                    break
    
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
    # Lister quelques fichiers disponibles pour debug
    try:
        sample_files = []
        for root, dirs, files in os.walk(DOCS_PATH):
            for file in files[:5]:  # Limiter à 5 fichiers
                if file.endswith('.pdf'):
                    sample_files.append(file)
        if sample_files:
            print(f"📄 Exemples de fichiers disponibles : {', '.join(sample_files)}")
    except:
        pass
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


@app.get("/status")
async def status():
    """
    Retourne l'état détaillé de l'API et de la base de données
    """
    import chromadb
    
    status_info = {
        "ok": True,
        "rag_initialized": rag_engine is not None,
        "environment": {
            "USE_API_EMBEDDINGS": os.getenv("USE_API_EMBEDDINGS", "non définie"),
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "non définie"),
            "HUGGINGFACE_API_KEY": "✅ configurée" if os.getenv("HUGGINGFACE_API_KEY") else "❌ non configurée",
            "CHROMA_DB_PATH": os.getenv("CHROMA_DB_PATH", "/tmp/chroma_db"),
            "AUTO_INGEST_ON_STARTUP": os.getenv("AUTO_INGEST_ON_STARTUP", "true"),
        },
        "chromadb": {},
        "docs_folder": {}
    }
    
    # Vérifier ChromaDB
    db_path = os.getenv("CHROMA_DB_PATH", "/tmp/chroma_db")
    try:
        db = chromadb.PersistentClient(path=db_path)
        collection = db.get_collection("renovation_knowledge")
        status_info["chromadb"] = {
            "path": db_path,
            "collection": "renovation_knowledge",
            "document_count": collection.count(),
            "status": "✅ connecté"
        }
    except Exception as e:
        status_info["chromadb"] = {
            "path": db_path,
            "status": f"❌ erreur: {str(e)}"
        }
    
    # Vérifier le dossier docs
    docs_path = os.path.join(BASE_DIR, "docs")
    try:
        if os.path.exists(docs_path):
            # Compter les fichiers PDF récursivement
            pdf_count = 0
            for root, dirs, files in os.walk(docs_path):
                pdf_count += len([f for f in files if f.endswith('.pdf')])
            
            status_info["docs_folder"] = {
                "path": docs_path,
                "exists": True,
                "pdf_count": pdf_count,
                "status": "✅ trouvé"
            }
        else:
            status_info["docs_folder"] = {
                "path": docs_path,
                "exists": False,
                "status": "❌ dossier non trouvé"
            }
    except Exception as e:
        status_info["docs_folder"] = {
            "path": docs_path,
            "status": f"❌ erreur: {str(e)}"
        }
    
    # Déterminer l'état global
    if not status_info["rag_initialized"]:
        status_info["ok"] = False
        status_info["message"] = "RAG non initialisé"
    elif status_info["chromadb"].get("document_count", 0) == 0:
        status_info["ok"] = False
        status_info["message"] = "ChromaDB est vide - les réponses seront 'Empty Response'"
    else:
        status_info["message"] = f"RAG opérationnel avec {status_info['chromadb']['document_count']} documents"
    
    return status_info


@app.post("/ingest")
async def trigger_ingest(force: bool = False, max_docs: int = 50):
    """
    Déclenche manuellement l'ingestion des documents
    
    Args:
        force: Si True, réindexe même si des documents existent
        max_docs: Nombre maximum de documents à indexer
    """
    try:
        # Importer et lancer l'ingestion
        ingestion_path = os.path.join(os.path.dirname(__file__), "..", "ingestion")
        sys.path.insert(0, ingestion_path)
        
        from ingest_api import ingest_documents
        
        # Exécuter dans un thread pour ne pas bloquer
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            success = await loop.run_in_executor(
                executor, 
                lambda: ingest_documents(force=force, max_docs=max_docs)
            )
        
        if success:
            return {"ok": True, "message": "Ingestion terminée avec succès"}
        else:
            return {"ok": False, "message": "Échec de l'ingestion"}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'ingestion: {str(e)}")

@app.post("/query", response_model=RAGResponse)
async def query_rag(request: RAGRequest):
    """
    Pose une question au système RAG et retourne une réponse avec sources
    """
    import time
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    global rag_engine
    
    if not rag_engine:
        raise HTTPException(status_code=503, detail="Moteur RAG non initialisé")
    
    try:
        print(f"📝 Requête RAG reçue : {request.question[:100]}...")
        
        # Construire la question personnalisée si des résultats DPE sont fournis
        question = request.question
        if request.dpe_results:
            # Personnaliser la question avec les résultats du DPE
            classe_dpe = request.dpe_results.get("classe_dpe_finale", "inconnue")
            etiquette_energie = request.dpe_results.get("etiquette_energie", "inconnue")
            
            question = f"""Mon logement a un DPE {classe_dpe} (étiquette énergétique {etiquette_energie}).
{request.question}

Peux-tu me donner des conseils personnalisés de rénovation énergétique adaptés à mon DPE ?"""
        
        print("🔍 Interrogation du RAG...")
        print(f"⏱️  Timeout maximal: 10 minutes (600 secondes)")
        start_time = time.time()
        
        # Exécuter la requête RAG dans un thread séparé pour éviter le conflit avec l'event loop
        # La méthode query() de RenovationRAG gère maintenant elle-même l'isolation de l'event loop
        loop = asyncio.get_event_loop()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Exécuter avec un timeout asyncio de 10 minutes
                response = await asyncio.wait_for(
                    loop.run_in_executor(executor, rag_engine.query, question),
                    timeout=600.0  # 10 minutes
                )
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            print(f"❌ Timeout après {elapsed_time:.2f} secondes")
            raise HTTPException(status_code=504, detail=f"La requête RAG a pris plus de 10 minutes ({elapsed_time:.2f}s)")
        except TimeoutError as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Timeout dans le thread RAG après {elapsed_time:.2f} secondes")
            raise HTTPException(status_code=504, detail=f"La requête RAG a pris plus de 10 minutes ({elapsed_time:.2f}s)")
        
        elapsed_time = time.time() - start_time
        print(f"✅ Réponse RAG obtenue en {elapsed_time:.2f} secondes")
        
        # Debug: afficher la structure de la réponse
        print(f"🔍 Type de réponse: {type(response)}")
        print(f"🔍 Attributs de la réponse: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
        # Extraire le texte de la réponse (streaming)
        # PRIORITÉ: Utiliser response.response directement (disponible même en mode streaming)
        texte_complet = ""
        if hasattr(response, 'response'):
            # L'attribut response contient le texte complet (même en mode streaming)
            texte_complet = str(response.response)
            print(f"✅ Texte extrait depuis response.response (longueur: {len(texte_complet)} caractères)")
            
            # LOGS DÉTAILLÉS - Analyser le contenu de la réponse
            print(f"📊 Analyse de la réponse:")
            print(f"   - Nombre de caractères: {len(texte_complet)}")
            print(f"   - Nombre de mots: {len(texte_complet.split())}")
            print(f"   - Nombre de lignes: {len(texte_complet.splitlines())}")
            
            # Vérifier si la réponse semble tronquée
            paragraphs = [p for p in texte_complet.split('\n\n') if p.strip()]
            print(f"   - Paragraphes détectés: {len(paragraphs)}")
            
            if len(paragraphs) > 0:
                print(f"   - Premier paragraphe (100 chars): {paragraphs[0][:100]}...")
                print(f"   - Dernier paragraphe (100 chars): {paragraphs[-1][:100]}...")
                
                # Vérifier si le dernier paragraphe se termine correctement
                last_para = paragraphs[-1].strip()
                ends_properly = last_para.endswith('.') or last_para.endswith('!') or last_para.endswith('?') or last_para.endswith(':')
                if not ends_properly and len(texte_complet) > 500:
                    print(f"   ⚠️  ALERTE: Le dernier paragraphe ne se termine pas par ponctuation - possible troncature!")
                    print(f"      Derniers 200 chars: ...{texte_complet[-200:]}")
            
            # Afficher un aperçu du texte complet
            print(f"📝 Aperçu complet de la réponse:")
            print(f"   Début (300 chars): {texte_complet[:300]}")
            if len(texte_complet) > 600:
                print(f"   ...")
                print(f"   Fin (300 chars): {texte_complet[-300:]}")
            else:
                print(f"   Fin: {texte_complet}")
        elif hasattr(response, 'response_gen'):
            # Mode streaming: collecter tous les tokens si response.response n'est pas disponible
            print("📡 Mode streaming détecté, collecte des tokens depuis response_gen...")
            token_count = 0
            try:
                for token in response.response_gen:
                    texte_complet += token
                    token_count += 1
                    if token_count <= 5:  # Afficher les 5 premiers tokens pour debug
                        print(f"   Token {token_count}: {repr(token[:50])}")
                print(f"📊 Total de {token_count} tokens collectés")
            except Exception as e:
                print(f"⚠️  Erreur lors de la collecte des tokens: {e}")
                import traceback
                traceback.print_exc()
        elif hasattr(response, 'response'):
            # Mode non-streaming: utiliser l'attribut response
            texte_complet = str(response.response)
        elif hasattr(response, 'get_response'):
            # Méthode alternative pour obtenir la réponse
            texte_complet = str(response.get_response())
        else:
            # Dernier recours: convertir en string
            texte_complet = str(response)
        
        # Vérifier que le texte n'est pas vide
        if not texte_complet or texte_complet.strip() == "":
            print("⚠️  La réponse RAG est vide, vérification de la structure de la réponse...")
            print(f"   Type de réponse: {type(response)}")
            print(f"   Attributs disponibles: {[attr for attr in dir(response) if not attr.startswith('_')]}")
            if hasattr(response, '__dict__'):
                print(f"   __dict__: {response.__dict__}")
            # Essayer d'autres méthodes pour extraire le texte
            if hasattr(response, 'response'):
                print("🔄 Tentative avec response.response...")
                texte_complet = str(response.response)
            elif hasattr(response, 'text'):
                print("🔄 Tentative avec response.text...")
                texte_complet = str(response.text)
            elif hasattr(response, 'answer'):
                print("🔄 Tentative avec response.answer...")
                texte_complet = str(response.answer)
            else:
                # Si toujours vide, utiliser une valeur par défaut
                texte_complet = "Empty Response"
                print("❌ Impossible d'extraire le texte de la réponse RAG")
                print(f"   Représentation de la réponse: {repr(response)}")
        
        print(f"📝 Texte final extrait (longueur: {len(texte_complet)} caractères)")
        
        # Vérification finale de la troncature
        if len(texte_complet) > 0:
            # Compter les sections/paragraphes numérotés
            import re
            numbered_sections = re.findall(r'^\d+\.\s+\*\*', texte_complet, re.MULTILINE)
            print(f"   - Sections numérotées détectées: {len(numbered_sections)}")
            
            # Chercher des patterns de troncature
            if texte_complet.strip().endswith('...') or texte_complet.strip().endswith('…'):
                print(f"   ⚠️  ALERTE: Le texte se termine par '...' - possible troncature!")
            
            # Vérifier la présence de balises de fin attendues
            if '[ANALYSE]' in texte_complet or '[SCENARIO' in texte_complet:
                if '[/ANALYSE]' not in texte_complet and '[/SCENARIO_1]' not in texte_complet and '[/SCENARIO_2]' not in texte_complet:
                    print(f"   ⚠️  ALERTE: Balises de structure non fermées - possible troncature!")
        
        if len(texte_complet) < 100:
            print(f"   Aperçu complet: {texte_complet[:200]}")
        
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
        pdf_filename = None
        try:
            outputs_dir = os.path.join(BASE_DIR, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            pdf_filename = f"rapport_renovation_{os.urandom(8).hex()}.pdf"
            pdf_path = os.path.join(outputs_dir, pdf_filename)
            
            # Construire le prompt complet pour parser les infos du bâtiment
            prompt_complet = question
            if request.dpe_results:
                # Ajouter les infos DPE au prompt pour le parsing
                dpe_info = f"""
DPE ACTUEL: {request.dpe_results.get('classe_dpe_finale', 'N/A')}
"""
                if 'surface_habitable_logement' in request.dpe_results:
                    dpe_info += f"Surface: {request.dpe_results.get('surface_habitable_logement')} m2\n"
                prompt_complet = dpe_info + prompt_complet
            
            # Parser les informations du bâtiment depuis la question et les résultats DPE
            building_info = parse_building_info(prompt_complet)
            # Enrichir avec les données DPE si disponibles (priorité aux données DPE réelles)
            if request.dpe_results:
                # Classe DPE actuelle
                if 'classe_dpe_finale' in request.dpe_results:
                    building_info['dpe_actuel'] = request.dpe_results.get('classe_dpe_finale', 'N/A')
                elif 'etiquette_energie' in request.dpe_results:
                    building_info['dpe_actuel'] = request.dpe_results.get('etiquette_energie', 'N/A')
                
                # Département (depuis code_departement_ban si disponible)
                if 'code_departement_ban' in request.dpe_results:
                    building_info['departement'] = str(request.dpe_results.get('code_departement_ban', 'N/A'))
                
                # Année de construction
                if 'annee_construction' in request.dpe_results:
                    building_info['annee'] = str(request.dpe_results.get('annee_construction', 'N/A'))
                
                # Surface
                if 'surface_habitable_logement' in request.dpe_results:
                    surface = request.dpe_results.get('surface_habitable_logement')
                    building_info['surface'] = f"{surface} m2" if surface else 'N/A'
                
                # Données techniques
                if 'ubat_w_par_m2_k' in request.dpe_results:
                    ubat = request.dpe_results.get('ubat_w_par_m2_k')
                    building_info['ubat'] = f"{ubat:.2f} W/m2.K" if ubat else 'N/A'
                
                if 'conso_chauffage_ep_par_m2' in request.dpe_results:
                    conso = request.dpe_results.get('conso_chauffage_ep_par_m2')
                    building_info['conso_chauffage'] = f"{conso:.1f} kWhEP/m2" if conso else 'N/A'
                
                if 'emission_ges_chauffage_par_m2' in request.dpe_results:
                    ges = request.dpe_results.get('emission_ges_chauffage_par_m2')
                    building_info['emissions_co2'] = f"{ges:.1f} kgCO2/m2" if ges else 'N/A'
            
            # Parser la réponse du RAG
            parsed_response = parse_rag_response(texte_complet)
            # Générer le PDF
            generate_renovation_pdf(building_info, parsed_response, pdf_path)
            print(f"✅ PDF généré avec succès: {pdf_filename}")
        except Exception as pdf_error:
            print(f"⚠️  Erreur lors de la génération du PDF: {pdf_error}")
            import traceback
            traceback.print_exc()
            # Ne pas bloquer la réponse si le PDF échoue
            pdf_filename = None
        
        return RAGResponse(
            ok=True,
            data={
                "response": texte_complet,
                "sources": sources,
                "pdf_filename": pdf_filename  # Nom du fichier PDF généré
            }
        )
        
    except Exception as e:
        print(f"❌ Erreur lors de la requête RAG: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la réponse: {str(e)}")

@app.get("/pdf/{filename}")
async def download_pdf(filename: str):
    """
    Télécharge un PDF généré précédemment
    """
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    pdf_path = os.path.join(outputs_dir, filename)
    
    # Sécurité: vérifier que le fichier est dans le dossier outputs
    if not os.path.abspath(pdf_path).startswith(os.path.abspath(outputs_dir)):
        raise HTTPException(status_code=403, detail="Accès non autorisé")
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF non trouvé: {filename}")
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.post("/query/pdf")
async def query_rag_and_generate_pdf(request: RAGRequest):
    """
    Pose une question au système RAG, génère une réponse et crée un PDF du rapport
    """
    import time
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    global rag_engine
    
    if not rag_engine:
        raise HTTPException(status_code=503, detail="Moteur RAG non initialisé")
    
    try:
        print(f"📝 Requête RAG (PDF) reçue : {request.question[:100]}...")
        
        # Construire la question personnalisée si des résultats DPE sont fournis
        question = request.question
        if request.dpe_results:
            # Personnaliser la question avec les résultats du DPE
            classe_dpe = request.dpe_results.get("classe_dpe_finale", "inconnue")
            etiquette_energie = request.dpe_results.get("etiquette_energie", "inconnue")
            
            question = f"""Mon logement a un DPE {classe_dpe} (étiquette énergétique {etiquette_energie}).
{request.question}

Peux-tu me donner des conseils personnalisés de rénovation énergétique adaptés à mon DPE ?"""
        
        print("🔍 Interrogation du RAG...")
        print(f"⏱️  Timeout maximal: 10 minutes (600 secondes)")
        start_time = time.time()
        
        # Exécuter la requête RAG dans un thread séparé pour éviter le conflit avec l'event loop
        # La méthode query() de RenovationRAG gère maintenant elle-même l'isolation de l'event loop
        loop = asyncio.get_event_loop()
        
        # Utiliser un timeout asyncio pour éviter que ça bloque indéfiniment
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Exécuter avec un timeout asyncio de 10 minutes
                response = await asyncio.wait_for(
                    loop.run_in_executor(executor, rag_engine.query, question),
                    timeout=600.0  # 10 minutes
                )
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            print(f"❌ Timeout après {elapsed_time:.2f} secondes")
            raise HTTPException(status_code=504, detail=f"La requête RAG a pris plus de 10 minutes ({elapsed_time:.2f}s)")
        except TimeoutError as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Timeout dans le thread RAG après {elapsed_time:.2f} secondes")
            raise HTTPException(status_code=504, detail=f"La requête RAG a pris plus de 10 minutes ({elapsed_time:.2f}s)")
        
        elapsed_time = time.time() - start_time
        print(f"✅ Réponse RAG obtenue en {elapsed_time:.2f} secondes")
        
        # Debug: afficher la structure de la réponse
        print(f"🔍 Type de réponse: {type(response)}")
        print(f"🔍 Attributs de la réponse: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
        # Extraire le texte de la réponse (streaming)
        # PRIORITÉ: Utiliser response.response directement (disponible même en mode streaming)
        texte_complet = ""
        if hasattr(response, 'response'):
            # L'attribut response contient le texte complet (même en mode streaming)
            texte_complet = str(response.response)
            print(f"✅ Texte extrait depuis response.response (longueur: {len(texte_complet)} caractères)")
        elif hasattr(response, 'response_gen'):
            # Mode streaming: collecter tous les tokens si response.response n'est pas disponible
            print("📡 Mode streaming détecté, collecte des tokens depuis response_gen...")
            token_count = 0
            try:
                for token in response.response_gen:
                    texte_complet += token
                    token_count += 1
                    if token_count <= 5:  # Afficher les 5 premiers tokens pour debug
                        print(f"   Token {token_count}: {repr(token[:50])}")
                print(f"📊 Total de {token_count} tokens collectés")
            except Exception as e:
                print(f"⚠️  Erreur lors de la collecte des tokens: {e}")
                import traceback
                traceback.print_exc()
        elif hasattr(response, 'response'):
            # Mode non-streaming: utiliser l'attribut response
            texte_complet = str(response.response)
        elif hasattr(response, 'get_response'):
            # Méthode alternative pour obtenir la réponse
            texte_complet = str(response.get_response())
        else:
            # Dernier recours: convertir en string
            texte_complet = str(response)
        
        # Vérifier que le texte n'est pas vide
        if not texte_complet or texte_complet.strip() == "":
            print("⚠️  La réponse RAG est vide, vérification de la structure de la réponse...")
            print(f"   Type de réponse: {type(response)}")
            print(f"   Attributs disponibles: {[attr for attr in dir(response) if not attr.startswith('_')]}")
            if hasattr(response, '__dict__'):
                print(f"   __dict__: {response.__dict__}")
            # Essayer d'autres méthodes pour extraire le texte
            if hasattr(response, 'response'):
                print("🔄 Tentative avec response.response...")
                texte_complet = str(response.response)
            elif hasattr(response, 'text'):
                print("🔄 Tentative avec response.text...")
                texte_complet = str(response.text)
            elif hasattr(response, 'answer'):
                print("🔄 Tentative avec response.answer...")
                texte_complet = str(response.answer)
            else:
                # Si toujours vide, utiliser une valeur par défaut
                texte_complet = "Empty Response"
                print("❌ Impossible d'extraire le texte de la réponse RAG")
                print(f"   Représentation de la réponse: {repr(response)}")
        
        print(f"📝 Texte final extrait (longueur: {len(texte_complet)} caractères)")
        
        # Vérification finale de la troncature
        if len(texte_complet) > 0:
            # Compter les sections/paragraphes numérotés
            import re
            numbered_sections = re.findall(r'^\d+\.\s+\*\*', texte_complet, re.MULTILINE)
            print(f"   - Sections numérotées détectées: {len(numbered_sections)}")
            
            # Chercher des patterns de troncature
            if texte_complet.strip().endswith('...') or texte_complet.strip().endswith('…'):
                print(f"   ⚠️  ALERTE: Le texte se termine par '...' - possible troncature!")
            
            # Vérifier la présence de balises de fin attendues
            if '[ANALYSE]' in texte_complet or '[SCENARIO' in texte_complet:
                if '[/ANALYSE]' not in texte_complet and '[/SCENARIO_1]' not in texte_complet and '[/SCENARIO_2]' not in texte_complet:
                    print(f"   ⚠️  ALERTE: Balises de structure non fermées - possible troncature!")
        
        if len(texte_complet) < 100:
            print(f"   Aperçu complet: {texte_complet[:200]}")
        
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
            # Construire le prompt complet pour parser les infos du bâtiment
            prompt_complet = question
            if request.dpe_results:
                # Ajouter les infos DPE au prompt pour le parsing
                dpe_info = f"""
DPE ACTUEL: {request.dpe_results.get('classe_dpe_finale', 'N/A')}
"""
                if 'surface_habitable_logement' in request.dpe_results:
                    dpe_info += f"Surface: {request.dpe_results.get('surface_habitable_logement')} m2\n"
                prompt_complet = dpe_info + prompt_complet
            
            # Parser les informations du bâtiment depuis la question et les résultats DPE
            building_info = parse_building_info(prompt_complet)
            # Enrichir avec les données DPE si disponibles (priorité aux données DPE réelles)
            if request.dpe_results:
                # Classe DPE actuelle
                if 'classe_dpe_finale' in request.dpe_results:
                    building_info['dpe_actuel'] = request.dpe_results.get('classe_dpe_finale', 'N/A')
                elif 'etiquette_energie' in request.dpe_results:
                    building_info['dpe_actuel'] = request.dpe_results.get('etiquette_energie', 'N/A')
                
                # Département (depuis code_departement_ban si disponible)
                if 'code_departement_ban' in request.dpe_results:
                    building_info['departement'] = str(request.dpe_results.get('code_departement_ban', 'N/A'))
                
                # Année de construction
                if 'annee_construction' in request.dpe_results:
                    building_info['annee'] = str(request.dpe_results.get('annee_construction', 'N/A'))
                
                # Surface
                if 'surface_habitable_logement' in request.dpe_results:
                    surface = request.dpe_results.get('surface_habitable_logement')
                    building_info['surface'] = f"{surface} m2" if surface else 'N/A'
                
                # Données techniques
                if 'ubat_w_par_m2_k' in request.dpe_results:
                    ubat = request.dpe_results.get('ubat_w_par_m2_k')
                    building_info['ubat'] = f"{ubat:.2f} W/m2.K" if ubat else 'N/A'
                
                if 'conso_chauffage_ep_par_m2' in request.dpe_results:
                    conso = request.dpe_results.get('conso_chauffage_ep_par_m2')
                    building_info['conso_chauffage'] = f"{conso:.1f} kWhEP/m2" if conso else 'N/A'
                
                if 'emission_ges_chauffage_par_m2' in request.dpe_results:
                    ges = request.dpe_results.get('emission_ges_chauffage_par_m2')
                    building_info['emissions_co2'] = f"{ges:.1f} kgCO2/m2" if ges else 'N/A'
            
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
    # Utiliser la variable d'environnement PORT pour Render, sinon 8002 par défaut
    port = int(os.getenv("PORT", "8002"))
    print(f"🚀 Démarrage de l'API sur le port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

