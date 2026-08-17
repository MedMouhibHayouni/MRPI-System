"""
Utilitaires de cache partagés entre les moteurs CF et CBF.

Centralise le calcul d'empreinte (fingerprint) des fichiers source,
pour que CF.py et CBF.py invalident leur cache pickle de la
même façon, sans dupliquer la logique de hash.

Aucune modification ici — audité, pas de bug, pas de code mort.
"""

import hashlib
import os
from typing import Any, Dict, List


def compute_source_fingerprint(paths: List[str]) -> str:
    """
    Calcule une empreinte SHA-256 basée sur le contenu des fichiers source.
    """
    hasher = hashlib.sha256()
    for path in sorted(paths):
        if not os.path.exists(path):
            hasher.update(f"MISSING:{path}".encode())
            continue
        hasher.update(path.encode())
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def is_cache_valid(model_data: Dict[str, Any], source_paths: List[str]) -> bool:
    """
    Vérifie qu'un dict de modèle (pickle chargé) correspond
    à l'état actuel des fichiers source.
    """
    current_fingerprint = compute_source_fingerprint(source_paths)
    return model_data.get("source_fingerprint") == current_fingerprint


def compute_content_fingerprint(*args: Any) -> str:
    """
    Calcule une empreinte SHA-256 à partir de données Python en mémoire.
    """
    import json

    payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
