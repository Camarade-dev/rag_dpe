import sys
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

# Ajout du chemin src pour trouver tes modules
sys.path.append("./src")
from rag_core.query_engine import RenovationRAG

def lire_prompt_externe(chemin_fichier):
    """Fonction utilitaire pour lire le contenu d'un fichier texte"""
    if not os.path.exists(chemin_fichier):
        print(f"❌ ERREUR CRITIQUE : Le fichier '{chemin_fichier}' est introuvable !")
        return None
    
    try:
        with open(chemin_fichier, "r", encoding="utf-8") as f:
            contenu = f.read()
        return contenu
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier : {e}")
        return None

def sauvegarder_resultats(texte_reponse, sources_info, dossier_sortie="outputs"):
    """Sauvegarde la réponse et les sources dans deux fichiers distincts"""
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
    
    # 1. Sauvegarde de la RÉPONSE seule
    chemin_reponse = os.path.join(dossier_sortie, "reponse_rag.txt")
    try:
        with open(chemin_reponse, "w", encoding="utf-8") as f:
            f.write(texte_reponse)
        print(f"\n💾 Réponse sauvegardée dans : {chemin_reponse}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde réponse : {e}")

    # 2. Sauvegarde des SOURCES seules
    chemin_sources = os.path.join(dossier_sortie, "sources_rag.txt")
    try:
        with open(chemin_sources, "w", encoding="utf-8") as f:
            f.write("=== DOCUMENTS UTILISÉS ===\n")
            f.write(sources_info)
        print(f"💾 Sources sauvegardées dans : {chemin_sources}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde sources : {e}")

def main():
    print("==================================================")
    print("🏠 ASSISTANT RÉNOVATION - MODE FICHIER & DUAL SAVE")
    print("==================================================")

    # 1. Initialisation
    rag = RenovationRAG()
    
    # 2. Lecture du prompt
    nom_fichier_entree = "prompts_rag.txt"
    dossier_prompts = "./prompts"
    chemin_entree = os.path.join(dossier_prompts, nom_fichier_entree)

    print(f"\n📂 Lecture du fichier d'entrée : {chemin_entree} ...")
    prompt_utilisateur = lire_prompt_externe(chemin_entree)

    # 3. Exécution
    if prompt_utilisateur:
        print("\n🚀 Contenu récupéré. Génération de la réponse...")
        print("--------------------------------------------------")
        
        # Envoi de la requête
        response = rag.query(prompt_utilisateur)
        
        # Captation du Streaming
        texte_complet = ""
        for token in response.response_gen:
            print(token, end="", flush=True)
            texte_complet += token
        
        print("\n\n--------------------------------------------------")
        
        # Extraction et Formatage des sources
        info_sources = ""
        print("🔍 SOURCES DOCUMENTAIRES UTILISÉES :")
        
        if response.source_nodes:
            # On parcourt chaque document trouvé
            for node in response.source_nodes:
                nom_pdf = node.metadata.get('file_name', 'Inconnu')
                page = node.metadata.get('page_label', '?')
                score = node.score if node.score else 0.0
                
                # Format de la ligne dans le fichier txt
                ligne_source = f"{nom_pdf} (Page {page}) - Score: {score:.2f}\n"
                
                print(f"   📄 {ligne_source.strip()}")
                info_sources += ligne_source
        else:
            msg = "Aucune source pertinente trouvée dans la base de données."
            print(f"   ⚠️ {msg}")
            info_sources = msg

        # 4. Sauvegarde dans les deux fichiers
        sauvegarder_resultats(texte_complet, info_sources)

if __name__ == "__main__":
    main()