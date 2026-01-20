#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour vérifier la génération de PDF avec des données DPE simulées.
Ce script simule une requête complète à l'API RAG et vérifie que le PDF est généré correctement.

Usage:
    python test_pdf_generation.py
    
    Ou avec une URL personnalisée:
    RAG_API_URL=http://localhost:8002 python test_pdf_generation.py
"""

import os
import sys
import requests
import json
from pathlib import Path

# Ajouter le chemin src pour les imports (si nécessaire pour les tests locaux)
BASE_DIR = Path(__file__).parent
if (BASE_DIR / "src").exists():
    sys.path.insert(0, str(BASE_DIR / "src"))

# Configuration
RAG_API_URL = os.getenv("RAG_API_URL", "RAG_API_URL=https://rag-dpe-1.onrender.com")
TEST_OUTPUT_DIR = BASE_DIR / "test_outputs"
TEST_OUTPUT_DIR.mkdir(exist_ok=True)


def simulate_dpe_data():
    """Génère des données DPE simulées complètes."""
    return {
        "classe_dpe_finale": "D",
        "etiquette_energie": "D",
        "etiquette_climat": "C",
        "code_departement_ban": 75,  # Paris
        "annee_construction": 1985,
        "surface_habitable_logement": 70,
        "hauteur_sous_plafond": 2.5,
        "nombre_niveau_logement": 2,
        "ubat_w_par_m2_k": 1.85,
        "conso_5_usages_par_m2_ef": 120.5,
        "conso_5_usages_par_m2_ep": 185.2,
        "conso_chauffage_ep_par_m2": 150.3,
        "conso_ecs_ep_par_m2": 25.1,
        "emission_ges_5_usages_par_m2": 35.2,
        "emission_ges_chauffage_par_m2": 28.5,
        "emission_ges_ecs_par_m2": 5.1,
        "score_ubat": 65,
        "score_chauffage_ep": 45,
        "score_ecs_ep": 70,
        "score_ges_chauffage": 50,
        "score_ges_ecs": 75
    }


def test_rag_query_with_pdf():
    """Teste l'API RAG avec génération de PDF."""
    print("=" * 60)
    print("🧪 TEST DE GÉNÉRATION DE PDF")
    print("=" * 60)
    
    # Données DPE simulées
    dpe_data = simulate_dpe_data()
    print(f"\n📊 Données DPE simulées:")
    print(f"   - Classe DPE: {dpe_data['classe_dpe_finale']}")
    print(f"   - Département: {dpe_data['code_departement_ban']}")
    print(f"   - Année: {dpe_data['annee_construction']}")
    print(f"   - Surface: {dpe_data['surface_habitable_logement']} m²")
    print(f"   - Ubat: {dpe_data['ubat_w_par_m2_k']} W/m².K")
    print(f"   - Conso chauffage: {dpe_data['conso_chauffage_ep_par_m2']} kWhEP/m²")
    print(f"   - Emissions CO2: {dpe_data['emission_ges_chauffage_par_m2']} kgCO2/m²")
    
    # Question pour le RAG
    question = """Quels sont les travaux de rénovation énergétique prioritaires pour mon logement ?
    Je souhaite améliorer ma classe DPE et réduire ma consommation d'énergie."""
    
    # Préparer la requête
    payload = {
        "question": question,
        "dpe_results": dpe_data
    }
    
    print(f"\n📤 Envoi de la requête à l'API RAG...")
    print(f"   URL: {RAG_API_URL}/query")
    print(f"   Question: {question[:100]}...")
    
    try:
        # Appel à l'API
        response = requests.post(
            f"{RAG_API_URL}/query",
            json=payload,
            timeout=600  # 10 minutes max
        )
        
        if response.status_code != 200:
            print(f"❌ Erreur HTTP {response.status_code}")
            print(f"   Réponse: {response.text[:500]}")
            return False
        
        result = response.json()
        
        if not result.get("ok"):
            print(f"❌ L'API a retourné une erreur")
            print(f"   Erreur: {result.get('error', 'Inconnue')}")
            return False
        
        data = result.get("data", {})
        rag_response = data.get("response", "")
        pdf_filename = data.get("pdf_filename")
        
        print(f"\n✅ Réponse RAG reçue")
        print(f"   Longueur de la réponse: {len(rag_response)} caractères")
        print(f"   Nombre de sources: {len(data.get('sources', []))}")
        
        if pdf_filename:
            print(f"\n📄 PDF généré: {pdf_filename}")
            
            # Télécharger le PDF
            pdf_url = f"{RAG_API_URL}/pdf/{pdf_filename}"
            print(f"   Téléchargement depuis: {pdf_url}")
            
            pdf_response = requests.get(pdf_url, timeout=30)
            
            if pdf_response.status_code == 200:
                # Sauvegarder le PDF localement
                pdf_path = TEST_OUTPUT_DIR / pdf_filename
                with open(pdf_path, "wb") as f:
                    f.write(pdf_response.content)
                
                print(f"   ✅ PDF téléchargé et sauvegardé: {pdf_path}")
                print(f"   Taille: {len(pdf_response.content)} octets")
                
                # Vérifier le contenu du PDF (basique)
                if len(pdf_response.content) > 1000:
                    print(f"   ✅ Le PDF semble valide (taille > 1 KB)")
                else:
                    print(f"   ⚠️  Le PDF semble trop petit, peut-être corrompu")
                
                # Analyser la réponse RAG pour vérifier les métriques
                print(f"\n🔍 Analyse de la réponse RAG pour les métriques...")
                check_metrics_in_response(rag_response)
                
                return True
            else:
                print(f"❌ Erreur lors du téléchargement du PDF: {pdf_response.status_code}")
                print(f"   Réponse: {pdf_response.text[:200]}")
                return False
        else:
            print(f"⚠️  Aucun PDF généré (pdf_filename manquant)")
            print(f"   Réponse complète: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Timeout lors de l'appel à l'API (plus de 10 minutes)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Impossible de se connecter à l'API")
        print(f"   Vérifiez que l'API est démarrée sur {RAG_API_URL}")
        print(f"   Vous pouvez la démarrer avec: python start_api.py")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_metrics_in_response(response_text):
    """Vérifie que la réponse contient des métriques financières (pas seulement N/A)."""
    print(f"   Recherche de métriques financières dans la réponse...")
    
    # Chercher des patterns de métriques
    metrics_found = {
        "coût": False,
        "aide": False,
        "économie": False,
        "rentabilité": False,
        "classe_visée": False
    }
    
    # Patterns pour détecter les métriques
    cost_patterns = [
        r'\d+\s*[kK]?\s*€',
        r'\d+\s*[kK]?\s*euros?',
        r'\d+\s*[kK]?\s*EUR',
        r'coût.*?\d+',
        r'prix.*?\d+',
        r'budget.*?\d+'
    ]
    
    aide_patterns = [
        r'aide.*?\d+',
        r'subvention.*?\d+',
        r'CEE.*?\d+',
        r'MaPrimeRenov.*?\d+',
        r'prime.*?\d+'
    ]
    
    econ_patterns = [
        r'économie.*?\d+',
        r'économies.*?\d+',
        r'épargne.*?\d+',
        r'gain.*?\d+'
    ]
    
    roi_patterns = [
        r'retour.*?\d+.*?ans?',
        r'rentabilité.*?\d+.*?ans?',
        r'amorti.*?\d+.*?ans?'
    ]
    
    classe_patterns = [
        r'classe.*?[A-G]',
        r'étiquette.*?[A-G]',
        r'visée.*?[A-G]'
    ]
    
    import re
    
    for pattern in cost_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            metrics_found["coût"] = True
            break
    
    for pattern in aide_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            metrics_found["aide"] = True
            break
    
    for pattern in econ_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            metrics_found["économie"] = True
            break
    
    for pattern in roi_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            metrics_found["rentabilité"] = True
            break
    
    for pattern in classe_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            metrics_found["classe_visée"] = True
            break
    
    # Afficher les résultats
    print(f"\n   📊 Métriques détectées:")
    for metric, found in metrics_found.items():
        status = "✅" if found else "❌"
        print(f"      {status} {metric.capitalize()}: {'Trouvé' if found else 'Non trouvé'}")
    
    # Vérifier s'il y a beaucoup de "N/A"
    na_count = response_text.count("N/A")
    if na_count > 5:
        print(f"\n   ⚠️  Attention: {na_count} occurrences de 'N/A' trouvées dans la réponse")
        print(f"      Cela peut indiquer que les métriques ne sont pas correctement extraites")
    else:
        print(f"\n   ✅ Peu d'occurrences de 'N/A' ({na_count}), c'est bon signe")
    
    # Chercher les balises structurées
    has_scenario_tags = "[SCENARIO_1]" in response_text or "[SCENARIO_2]" in response_text
    if has_scenario_tags:
        print(f"   ✅ Balises structurées [SCENARIO_1] ou [SCENARIO_2] trouvées")
    else:
        print(f"   ⚠️  Aucune balise structurée [SCENARIO_X] trouvée")
        print(f"      Le parser devra extraire depuis le texte libre")


def main():
    """Fonction principale."""
    print("\n" + "=" * 60)
    print("🚀 SCRIPT DE TEST - GÉNÉRATION DE PDF")
    print("=" * 60)
    print(f"\n📂 Répertoire de sortie: {TEST_OUTPUT_DIR}")
    print(f"🌐 URL de l'API RAG: {RAG_API_URL}")
    
    # Vérifier que l'API est accessible
    try:
        health_response = requests.get(f"{RAG_API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"\n✅ API accessible")
            print(f"   Status: {health_data.get('status')}")
            print(f"   RAG initialisé: {health_data.get('rag_initialized', False)}")
            if not health_data.get('rag_initialized', False):
                print(f"   ⚠️  Le RAG n'est pas encore initialisé, l'appel peut prendre plus de temps")
        else:
            print(f"\n⚠️  L'API répond mais le health check a retourné {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Impossible de se connecter à l'API sur {RAG_API_URL}")
        print(f"   Assurez-vous que l'API est démarrée:")
        print(f"   - En local: python start_api.py")
        print(f"   - Ou vérifiez que RAG_API_URL est correctement configuré")
        return
    except Exception as e:
        print(f"\n⚠️  Erreur lors du health check: {e}")
        print(f"   Continuons quand même...")
    
    # Lancer le test
    success = test_rag_query_with_pdf()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST RÉUSSI")
        print(f"📁 Le PDF a été sauvegardé dans: {TEST_OUTPUT_DIR}")
    else:
        print("❌ TEST ÉCHOUÉ")
        print("   Vérifiez les erreurs ci-dessus")
    print("=" * 60)


if __name__ == "__main__":
    main()
