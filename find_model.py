"""
Script utilitaire pour trouver et configurer le modèle LLM
"""
import os
import glob

MODELS_DIR = "./data/llm_models"

def find_gguf_models():
    """Trouve tous les fichiers .gguf dans le dossier des modèles"""
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Le dossier {MODELS_DIR} n'existe pas encore.")
        print(f"📁 Création du dossier...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"✅ Dossier créé. Placez votre fichier .gguf dans : {os.path.abspath(MODELS_DIR)}")
        return []
    
    pattern = os.path.join(MODELS_DIR, "*.gguf")
    models = glob.glob(pattern)
    return models

def main():
    print("=" * 60)
    print("🔍 Recherche des modèles LLM")
    print("=" * 60)
    print()
    
    models = find_gguf_models()
    
    if not models:
        print(f"⚠️  Aucun fichier .gguf trouvé dans : {os.path.abspath(MODELS_DIR)}")
        print()
        print("📋 Instructions :")
        print(f"   1. Téléchargez un modèle au format .gguf (ex: Llama, Mistral)")
        print(f"   2. Placez-le dans : {os.path.abspath(MODELS_DIR)}")
        print(f"   3. Relancez ce script pour vérifier")
        print()
        print("💡 Modèles recommandés :")
        print("   - Mistral 7B Instruct (Q4_K_M) : ~4 Go")
        print("   - Llama 3 8B Instruct (Q4_K_M) : ~4.5 Go")
        print("   - Phi-3 Mini (Q4_K_M) : ~2.5 Go")
        return
    
    print(f"✅ {len(models)} modèle(s) trouvé(s) :\n")
    for i, model_path in enumerate(models, 1):
        model_name = os.path.basename(model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        size_gb = size_mb / 1024
        
        print(f"   {i}. {model_name}")
        print(f"      📍 Chemin : {os.path.abspath(model_path)}")
        print(f"      💾 Taille : {size_gb:.2f} Go ({size_mb:.0f} Mo)")
        print()
    
    if len(models) == 1:
        print("✅ Un seul modèle trouvé. Il sera utilisé automatiquement.")
        print(f"   Si le code ne le trouve pas, vérifiez le nom dans query_engine.py")
    else:
        print("⚠️  Plusieurs modèles trouvés.")
        print("   Modifiez MODEL_PATH dans src/rag_core/query_engine.py pour choisir lequel utiliser.")

if __name__ == "__main__":
    main()

