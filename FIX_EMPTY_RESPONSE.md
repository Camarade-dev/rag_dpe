# 🔧 Solution au problème "Empty Response" sur Render

## 📋 Diagnostic du problème

Le problème "Empty Response" était causé par une **base ChromaDB vide** sur Render :

1. **`/tmp/chroma_db` est éphémère** - Ce dossier est effacé à chaque redéploiement sur Render
2. **Les documents n'étaient jamais indexés** - Le script d'ingestion n'était pas exécuté
3. **Sans contexte, le LLM renvoie "Empty Response"** - Normal car il n'y a rien à consulter

## ✅ Solution implémentée

### 1. Script d'ingestion compatible API (`src/ingestion/ingest_api.py`)

Un nouveau script d'ingestion qui :
- Utilise les **mêmes embeddings API** que l'API RAG (pas de torch/sentence-transformers)
- Fonctionne sur Render avec les contraintes de mémoire (512 MB)
- S'intègre automatiquement au démarrage de l'API

### 2. Ingestion automatique au démarrage

L'API vérifie maintenant au démarrage si ChromaDB est vide et lance l'ingestion automatiquement :
- Variable `AUTO_INGEST_ON_STARTUP=true` (activée par défaut)
- Variable `INGESTION_MAX_DOCS=100` pour limiter le nombre de documents au premier démarrage

### 3. Nouveaux endpoints de diagnostic

- `GET /status` - État détaillé de l'API, ChromaDB et du dossier docs
- `POST /ingest` - Déclencher manuellement l'ingestion

## 🚀 Configuration sur Render

### Variables d'environnement requises

```
HUGGINGFACE_API_KEY=hf_votre_cle_api
LLM_PROVIDER=huggingface
HUGGINGFACE_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
USE_API_EMBEDDINGS=true
CHROMA_DB_PATH=/tmp/chroma_db
AUTO_INGEST_ON_STARTUP=true
INGESTION_MAX_DOCS=100
PORT=8002
PYTHON_VERSION=3.12.0
```

### Build Command

```bash
pip install --upgrade pip setuptools wheel && pip install --no-cache-dir --prefer-binary -r requirements_render.txt
```

### Start Command

```bash
uvicorn src.api.main_api:app --host 0.0.0.0 --port $PORT
```

## 🔍 Vérification après déploiement

1. **Vérifier l'état** : `GET https://votre-app.onrender.com/status`
   
   Vous devriez voir :
   ```json
   {
     "ok": true,
     "rag_initialized": true,
     "chromadb": {
       "document_count": 50,
       "status": "✅ connecté"
     },
     "message": "RAG opérationnel avec 50 documents"
   }
   ```

2. **Tester une requête** : `POST https://votre-app.onrender.com/query`
   ```json
   {
     "question": "Quels travaux de rénovation pour améliorer mon DPE ?"
   }
   ```

## ⚠️ Points importants

### Temps de démarrage
Le premier démarrage sera plus long car l'ingestion des documents prend du temps :
- ~50 documents : 2-5 minutes
- ~100 documents : 5-10 minutes
- ~300 documents : 15-30 minutes

### Limitation de mémoire
Sur le plan gratuit Render (512 MB), limitez l'ingestion à 50-100 documents pour éviter les problèmes de mémoire.

### Persistance
⚠️ **Important** : `/tmp/chroma_db` n'est PAS persistant sur Render. Les documents seront réindexés à chaque redéploiement.

Pour une solution persistante, envisagez :
- **Render Disk** (payant) - Stockage persistant
- **Pinecone** (gratuit tier) - Vector store cloud
- **Qdrant Cloud** (gratuit tier) - Alternative à Pinecone

## 📊 Logs attendus au démarrage

```
============================================================
🔧 Initialisation du moteur RAG...
============================================================
📊 USE_API_EMBEDDINGS=true
📊 LLM_PROVIDER=huggingface
📊 HUGGINGFACE_API_KEY=✅ configurée
✅ torch n'est pas installé - bonne configuration pour économiser la RAM

🔄 Vérification de l'ingestion des documents...
🔍 Vérification de la base ChromaDB : /tmp/chroma_db
📊 Collection 'renovation_knowledge' contient 0 documents

============================================================
⚠️ COLLECTION VIDE - LANCEMENT DE L'INGESTION AUTOMATIQUE
============================================================
📄 Ingestion limitée à 100 documents maximum
🧠 Initialisation des embeddings via API Hugging Face...
✅ Embeddings initialisés
📏 Configuration du chunking...
💾 Connexion à ChromaDB...
⏳ Lecture des fichiers PDF...
📄 50 pages chargées en 2.5s
⚙️ Création des vecteurs (embeddings via API)...

✅ Indexation terminée en 180.0s
📊 150 chunks indexés dans ChromaDB
✅ Ingestion automatique terminée avec succès

🚀 Démarrage de l'initialisation du RAG...
✅ Moteur RAG prêt à l'emploi !
============================================================
✅ Moteur RAG prêt !
============================================================
🌐 L'API est prête à recevoir des requêtes sur le port $PORT
```

## 🔧 Dépannage

### "Empty Response" persiste
1. Vérifiez `/status` pour voir si ChromaDB contient des documents
2. Si `document_count: 0`, déclenchez manuellement : `POST /ingest`
3. Vérifiez les logs Render pour les erreurs d'ingestion

### Erreur de mémoire
1. Réduisez `INGESTION_MAX_DOCS` à 30-50
2. Redéployez l'application

### Timeout au démarrage
Render a un timeout de 30 minutes par défaut. L'ingestion devrait terminer dans ce délai pour 100 documents.
