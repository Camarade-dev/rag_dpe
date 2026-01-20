# Script de Test - Génération de PDF

Ce script permet de tester la génération de PDF avec des données DPE simulées.

## Prérequis

1. L'API RAG doit être démarrée (localement ou sur Render)
2. Python 3.7+ avec les packages `requests` installé

## Installation

```bash
pip install requests
```

## Utilisation

### Test en local

1. Démarrer l'API RAG :
```bash
python start_api.py
```

2. Dans un autre terminal, lancer le test :
```bash
cd DPE_IA_ia/RAG
python test_pdf_generation.py
```

### Test avec API distante (Render)

```bash
RAG_API_URL=https://rag-dpe-1.onrender.com python test_pdf_generation.py
```

## Ce que fait le script

1. ✅ Simule des données DPE complètes (classe D, Paris, 1985, 70m², etc.)
2. ✅ Envoie une requête à l'API RAG avec ces données
3. ✅ Vérifie que la réponse contient des métriques financières (coûts, aides, économies)
4. ✅ Télécharge le PDF généré
5. ✅ Sauvegarde le PDF dans `test_outputs/`
6. ✅ Analyse la réponse pour détecter les métriques (coût, aide, économies, rentabilité)

## Résultat attendu

Le script devrait :
- ✅ Générer un PDF avec succès
- ✅ Détecter des métriques financières dans la réponse (pas seulement "N/A")
- ✅ Sauvegarder le PDF dans `test_outputs/`

## Vérification manuelle

Après l'exécution, vous pouvez :
1. Ouvrir le PDF dans `test_outputs/` pour vérifier visuellement
2. Vérifier que les sections "Scenario 1" et "Scenario 2" contiennent :
   - Des coûts estimés (pas "N/A")
   - Des aides financières (pas "N/A")
   - Des économies annuelles (pas "N/A")
   - Des retours sur investissement (pas "N/A")

## Dépannage

**Erreur de connexion** :
- Vérifiez que l'API est démarrée
- Vérifiez l'URL avec `RAG_API_URL`

**Timeout** :
- L'initialisation du RAG peut prendre plusieurs minutes
- Augmentez le timeout dans le script si nécessaire

**PDF avec "N/A"** :
- Le LLM n'a peut-être pas généré les balises structurées
- Le parser devrait extraire depuis le texte libre
- Vérifiez les logs de l'API pour voir la réponse complète
