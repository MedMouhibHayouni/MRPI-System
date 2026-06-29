"""
Configuration centralisée pour le module de Filtrage Collaboratif (CF).

Toute constante partagée entre CF.py, KNN.py et SVD.py vit ici.
Une seule source de vérité — aucune duplication entre fichiers.
"""

# Configuration SVD
SVD_TEST_K = 5                      # Composantes pour le test de variance
SVD_VARIANCE_THRESHOLD = 0.50       # Seuil d'activation SVD (variance expliquée)
SVD_COMPONENTS = 10                 # Composantes si SVD activée

# Configuration KNN
KNN_NEIGHBORS = 3                   # Voisins utilisés pour les recommandations
MAX_NEIGHBORS = 20                  # Voisins max indexés au fit()