# Vérification du problème de mémoire sur Render

## Diagnostic

D'après les logs, le problème "Out of memory" se produit **avant** l'initialisation des embeddings. Cela suggère que :

1. **Le LLM se charge** (ligne 113 dans les logs)
2. **ChromaDB ou une autre dépendance** charge torch/onnxruntime
3. **Out of memory** avant même d'arriver aux embeddings

## Solution immédiate : Vérifier USE_API_EMBEDDINGS

**CRITIQUE** : Vérifiez que `USE_API_EMBEDDINGS=true` est bien configuré dans Render.

Dans les nouveaux logs, vous devriez voir :
```
📊 Variables d'environnement : USE_API_EMBEDDINGS=true
```

Si vous voyez `USE_API_EMBEDDINGS=non définie` ou `USE_API_EMBEDDINGS=false`, la variable n'est pas configurée.

## Actions à faire

1. **Vérifier la variable d'environnement** dans Render Dashboard
2. **Redéployer** après avoir ajouté/modifié la variable
3. **Vérifier les logs** pour voir :
   - `📊 Variables d'environnement : USE_API_EMBEDDINGS=...`
   - `🔍 Configuration embeddings : USE_API_EMBEDDINGS=...`
   - `✅ Embeddings via API Hugging Face...`

## Si le problème persiste

Le problème peut aussi venir de :
- ChromaDB qui charge onnxruntime
- `llama-index-llms-huggingface` qui installe torch même pour l'API
- Autres dépendances lourdes

Dans ce cas, il faudra peut-être :
- Utiliser un modèle plus léger
- Upgrader vers un plan Render payant
- Utiliser un autre provider (OpenAI, Anthropic) qui est plus léger
