import sys

print("--- Vérification de l'installation ---")

try:
    import llama_index.core
    print(f"✅ LlamaIndex Core version: {llama_index.core.__version__}")
except ImportError as e:
    print(f"❌ Erreur LlamaIndex: {e}")

try:
    import llama_cpp
    print(f"✅ Llama CPP version: {llama_cpp.__version__}")
except ImportError as e:
    print(f"❌ Erreur Llama CPP: {e}")

try:
    import chromadb
    print(f"✅ ChromaDB version: {chromadb.__version__}")
except ImportError as e:
    print(f"❌ Erreur ChromaDB: {e}")

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"   CUDA disponible (GPU) ? : {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Erreur PyTorch: {e}")

print("--------------------------------------")