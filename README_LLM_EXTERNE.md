# Configuration LLM Externe pour la RAG

Ce guide explique comment configurer l'API RAG pour utiliser un LLM externe au lieu d'un modèle local.

## Options disponibles

### 1. OpenAI (Recommandé pour la production) ⭐

**Avantages** : Rapide, fiable, bon support français

**Configuration** :
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-votre-cle-api
OPENAI_MODEL=gpt-3.5-turbo  # ou gpt-4, gpt-4-turbo-preview
```

**Coût** : ~$0.002 par requête (gpt-3.5-turbo)

### 2. Anthropic Claude

**Avantages** : Excellent pour le français, très performant

**Configuration** :
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-votre-cle-api
ANTHROPIC_MODEL=claude-3-haiku-20240307  # ou claude-3-sonnet-20240229
```

**Coût** : ~$0.00025 par requête (Claude Haiku)

### 3. Hugging Face Inference API

**Avantages** : Gratuit avec limitations, bon pour le développement

**Configuration** :
```bash
LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=hf_votre-cle-api
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

**Coût** : Gratuit jusqu'à un certain quota, puis payant

### 4. Ollama (Self-hosted)

**Avantages** : Gratuit, contrôle total, pas de limite

**Configuration** :
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434  # ou URL de votre serveur Ollama
OLLAMA_MODEL=mistral  # ou llama2, codellama, etc.
```

**Coût** : Gratuit (nécessite votre propre serveur)

## Configuration sur Render

### Variables d'environnement à ajouter dans Render

1. Allez sur votre service RAG API dans Render
2. Cliquez sur "Environment"
3. Ajoutez les variables suivantes :

**Pour OpenAI** :
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-votre-cle-api
OPENAI_MODEL=gpt-3.5-turbo
```

**Pour Anthropic** :
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-votre-cle-api
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

**Pour Hugging Face** :
```
LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=hf_votre-cle-api
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

### Mise à jour du render.yaml

Vous pouvez aussi ajouter ces variables dans `render.yaml` :

```yaml
  # API Python RAG
  - type: web
    name: rag-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.main_api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PORT
        value: 8002
      - key: CHROMA_DB_PATH
        value: /tmp/chroma_db
      - key: LLM_PROVIDER
        value: openai
      - key: OPENAI_API_KEY
        sync: false  # À configurer manuellement dans Render
      - key: OPENAI_MODEL
        value: gpt-3.5-turbo
```

## Obtenir une clé API

### OpenAI
1. Allez sur https://platform.openai.com/api-keys
2. Créez un compte ou connectez-vous
3. Créez une nouvelle clé API
4. Ajoutez des crédits (minimum $5)

### Anthropic
1. Allez sur https://console.anthropic.com/
2. Créez un compte
3. Allez dans "API Keys"
4. Créez une nouvelle clé

### Hugging Face
1. Allez sur https://huggingface.co/settings/tokens
2. Créez un compte ou connectez-vous
3. Créez un nouveau token avec les permissions "Read"
4. Pour l'Inference API, vous devrez peut-être activer le paiement (gratuit jusqu'à un certain quota)

## Test de la configuration

Après avoir configuré les variables d'environnement, redéployez le service et vérifiez les logs. Vous devriez voir :

```
🤖 Utilisation d'OpenAI : gpt-3.5-turbo
✅ LLM initialisé avec succès
```

## Fallback vers modèle local

Si aucun `LLM_PROVIDER` n'est configuré ou si la configuration est incorrecte, le système utilisera automatiquement le modèle local (`LlamaCPP`) s'il est disponible dans `./data/llm_models/`.

## Dépannage

**Erreur "OPENAI_API_KEY non définie"** :
- Vérifiez que la variable d'environnement est bien définie dans Render
- Redéployez le service après avoir ajouté la variable

**Erreur "Module not found"** :
- Vérifiez que `requirements.txt` contient les dépendances nécessaires
- Redéployez le service pour installer les nouvelles dépendances

**Réponses lentes** :
- Essayez un modèle plus rapide (gpt-3.5-turbo au lieu de gpt-4)
- Vérifiez votre connexion internet
- Pour Hugging Face, l'API peut être lente selon la charge

