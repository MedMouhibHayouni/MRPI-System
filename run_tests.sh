#!/usr/bin/env bash
# run_tests.sh — Lance la suite de tests unitaires + intégration avec
# mesure de couverture, et échoue (exit code != 0) si la couverture
# totale de src/ passe sous 70%.
#
# Usage :
#   ./run_tests.sh                # tests unitaires + intégration + couverture
#   ./run_tests.sh -k cbf         # ne lance que les tests dont le nom contient "cbf"
#   ./run_tests.sh tests/unit     # ne lance que le dossier unit/
#
# Le seuil de couverture (70%) est défini dans pytest.ini
# (--cov-fail-under=70), pas ici, pour rester la source unique de vérité.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "→ Aucun venv trouvé, création..."
    python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "→ Installation des dépendances de test..."
pip install --quiet -r requirements-test.txt

echo "→ Exécution de pytest --cov (seuil minimum : 70%)..."
set +e
pytest "$@"
STATUS=$?
set -e

if [ $STATUS -eq 0 ]; then
    echo ""
    echo "✅ Suite de tests OK — couverture >= 70% (voir htmlcov/index.html pour le détail)."
else
    echo ""
    echo "❌ Échec : soit un test a échoué, soit la couverture est sous 70%."
fi

exit $STATUS
