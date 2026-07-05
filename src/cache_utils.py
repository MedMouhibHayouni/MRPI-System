"""
Utilitaires de cache partagés entre les moteurs CF et CBF.

Centralise le calcul d'empreinte (fingerprint) des fichiers source,
pour que CF.py et CBF.py invalident leur cache pickle de la
même façon, sans dupliquer la logique de hash.
"""

import hashlib
import os
from typing import Any, Dict, List
import json


def compute_source_fingerprint(paths: List[str]) -> str:
    """
    Calcule une empreinte SHA-256 unique représentant l'état actuel d'un
    ensemble de fichiers source (contenu binaire, pas seulement le nom ou
    la date de modification).

    Pour chaque chemin, dans un ordre trié (déterministe, indépendant de
    l'ordre d'appel) : si le fichier n'existe pas, incorpore le marqueur
    "MISSING:<chemin>" dans le hash au lieu de lever une exception — un
    fichier source absent produit donc une empreinte différente d'un
    fichier présent, sans crasher l'appelant. Si le fichier existe, incorpore
    à la fois le chemin et le contenu brut du fichier dans le hash.

    Utilisée comme signature pour détecter si des fichiers de données
    (interactions.csv, resources.csv) ont changé depuis la dernière
    construction d'un modèle mis en cache (CF, CBF).

    Parameters
    ----------
    paths : List[str]
        Chemins absolus ou relatifs des fichiers à empreindre.

    Returns
    -------
    str
        Empreinte hexadécimale SHA-256 (64 caractères).
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
    Détermine si un modèle chargé depuis un pickle est encore synchronisé
    avec l'état actuel des fichiers source dont il dépend.

    Recalcule l'empreinte courante des fichiers source via
    compute_source_fingerprint() et la compare à l'empreinte stockée dans
    model_data au moment de la construction du modèle (clé
    "source_fingerprint"). Si la clé est absente de model_data, la
    comparaison échoue naturellement (None != empreinte calculée) et la
    fonction retourne False — un modèle sans fingerprint est donc
    toujours considéré comme périmé.

    Parameters
    ----------
    model_data : Dict[str, Any]
        Dictionnaire de modèle chargé depuis un pickle (CF ou CBF),
        censé contenir la clé "source_fingerprint".
    source_paths : List[str]
        Fichiers source à vérifier contre cette empreinte.

    Returns
    -------
    bool
        True si le modèle est toujours valide (aucun fichier source
        modifié depuis sa construction), False sinon.
    """

    current_fingerprint = compute_source_fingerprint(source_paths)
    return model_data.get("source_fingerprint") == current_fingerprint


def compute_content_fingerprint(*args: Any) -> str:
    """
    Calcule une empreinte SHA-256 déterministe à partir de données Python
    arbitraires passées en argument (dicts, listes, primitives), sans
    dépendre de fichiers sur disque.

    Sérialise les arguments en JSON avec des clés triées
    (sort_keys=True) pour garantir que deux appels avec un contenu
    sémantiquement identique mais un ordre de clés différent produisent
    la même empreinte. `default=str` assure que les objets non
    sérialisables nativement (ex: objets pandas) sont convertis en
    chaîne plutôt que de lever une exception.

    Utilisée par LLM_API.py pour générer les clés de cache des appels
    LLM (explanation, wording, study_plan) — deux requêtes avec un
    profil et des recommandations identiques doivent produire la même
    clé de cache, indépendamment de l'ordre d'insertion des champs.

    Parameters
    ----------
    *args : Any
        Données arbitraires à empreindre (nombre variable d'arguments).

    Returns
    -------
    str
        Empreinte hexadécimale SHA-256 (64 caractères).
    """

    payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
