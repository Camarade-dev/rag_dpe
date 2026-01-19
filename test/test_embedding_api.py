#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Hugging Face Inference Providers (router.huggingface.co)
Format correct: https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction
"""

import os
import sys
import json
import requests

# Configuration
MODEL_NAME = "intfloat/multilingual-e5-base"
# URL correcte pour embeddings via router.huggingface.co
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}/pipeline/feature-extraction"

def ensure_e5_prefix(text: str) -> str:
    """
    Pour E5, il est recommandé de préfixer avec 'query: ' ou 'passage: '.
    """
    t = text.strip()
    if not (t.startswith("query:") or t.startswith("passage:")):
        return f"query: {t}"
    return t

def query(payload, token):
    """Fonction pour faire une requête à l'API"""
    headers = {
        "Authorization": f"Bearer {token}",
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

def main():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        print("❌ Token manquant. Définis HF_TOKEN ou HUGGINGFACE_API_KEY.")
        print('   Exemple: export HUGGINGFACE_API_KEY="hf_..."')
        sys.exit(1)

    test_text = "Quels sont les travaux de rénovation énergétique pour un logement DPE C ?"
    text_prefixed = ensure_e5_prefix(test_text)

    print("🧪 Test Embeddings via router.huggingface.co")
    print("=" * 60)
    print(f"🤖 Modèle: {MODEL_NAME}")
    print(f"📝 Texte de test: {test_text}")
    print(f"📝 Avec préfixe: {text_prefixed}")
    print(f"🌐 URL: {API_URL}")
    print("=" * 60)

    payload = {
        "inputs": text_prefixed
    }

    print(f"\n📦 Payload: {json.dumps(payload, indent=2)}")
    print(f"⏳ Envoi de la requête...\n")

    try:
        response = query(payload, token)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            output = response.json()
            
            # Parser la réponse
            if isinstance(output, list):
                if len(output) > 0:
                    embedding = output[0] if isinstance(output[0], list) else output
                    print(f"✅ Embedding obtenu avec succès !")
                    print(f"📏 Dimension: {len(embedding)}")
                    print(f"📊 5 premiers éléments: {embedding[:5]}")
                    print(f"📊 5 derniers éléments: {embedding[-5:]}")
                    print("\n✅ La nouvelle API fonctionne correctement !")
                    sys.exit(0)
                else:
                    print(f"❌ Réponse vide")
                    sys.exit(1)
            else:
                print(f"📦 Format de réponse: {type(output).__name__}")
                print(f"📄 Réponse complète: {json.dumps(output, indent=2)[:500]}...")
                sys.exit(1)
                
        else:
            print(f"❌ Erreur {response.status_code}")
            try:
                error_detail = response.json()
                print(f"📄 Détails: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"📄 Réponse: {response.text[:500]}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
