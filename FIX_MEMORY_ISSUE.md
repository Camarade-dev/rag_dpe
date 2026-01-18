# Solution au problème "Out of memory" sur Render

## Diagnostic

Le problème persiste car :
1. Les packages `llama-index-llms-huggingface` ou `llama-index-embeddings-huggingface` peuvent installer `torch` comme dépendance optionnelle
2. L'import de `HuggingFaceEmbedding` au début du fichier peut déclencher le chargement de torch
3. ChromaDB peut aussi charger onnxruntime qui est lourd

## Solutions supplémentaires

### Solution A : Exclure explicitement torch (RECOMMANDÉ)

Modifier le buildCommand dans render.yaml pour exclure torch :

```yaml
buildCommand: |
  pip install --upgrade pip setuptools wheel
  pip install --no-cache-dir --prefer-binary -r requirements_render.txt
  # Forcer la désinstallation de torch si installé par erreur
  pip uninstall -y torch torchvision torchaudio || true
  pip cache purge
```

### Solution B : Utiliser un modèle d'embedding plus léger ou via API uniquement

Le problème peut aussi venir du fait que même avec l'API, certains packages dépendent de torch.

### Solution C : Vérifier que USE_API_EMBEDDINGS est bien à "true"

Dans les logs, vous devriez voir :
```
🔍 Configuration embeddings : USE_API_EMBEDDINGS=true
✅ Embeddings via API Hugging Face (pas de modèle en mémoire, économise ~400 MB RAM)
```

Si vous ne voyez pas ces messages, c'est que la variable n'est pas configurée correctement.

### Solution D : Upgrader vers un plan Render payant (si budget disponible)

- Starter Plan : $7/mois - 512 MB RAM (pas de cold start)
- Standard Plan : $25/mois - 2 GB RAM

## Vérifications immédiates

1. **Vérifier USE_API_EMBEDDINGS dans Render** :
   - Dashboard → rag-api → Environment
   - Vérifier que `USE_API_EMBEDDINGS` = `true` (pas `True` ou `"true"`)

2. **Vérifier les logs** :
   - Chercher le message `🔍 Configuration embeddings : USE_API_EMBEDDINGS=...`
   - Si vous voyez `USE_API_EMBEDDINGS=false`, la variable n'est pas configurée

3. **Vérifier requirements_render.txt** :
   - S'assurer que le build utilise bien `requirements_render.txt`
   - Dans les logs de build, vérifier qu'il n'installe pas torch
