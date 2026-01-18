#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour démarrer l'API RAG
"""
import os
import sys
import subprocess

def main():
    print("=" * 40)
    print("🚀 Démarrage de l'API RAG")
    print("=" * 40)
    print()
    
    # Vérifier si on est dans le bon répertoire
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Chemin vers l'API
    api_path = os.path.join("src", "api", "main_api.py")
    
    if not os.path.exists(api_path):
        print(f"❌ Erreur : Le fichier {api_path} est introuvable")
        print(f"   Répertoire actuel : {os.getcwd()}")
        sys.exit(1)
    
    print(f"📂 Répertoire : {os.getcwd()}")
    print(f"📄 Lancement : {api_path}")
    print()
    
    # Lancer l'API avec uvicorn
    try:
        # Essayer avec uvicorn d'abord
        cmd = [sys.executable, "-m", "uvicorn", "src.api.main_api:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
        subprocess.run(cmd, cwd=script_dir)
    except KeyboardInterrupt:
        print("\n\n⚠️  Arrêt de l'API...")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage : {e}")
        print("\n💡 Essayez de lancer directement :")
        print(f"   python {api_path}")

if __name__ == "__main__":
    main()
