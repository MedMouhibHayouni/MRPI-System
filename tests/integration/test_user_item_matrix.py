"""
Test d'intégration : user_itemMatrix.py lit réellement interactions.csv
et resources.csv depuis data/ et construit la matrice utilisateur-ressource.
Aucun mock ici — objectif : valider le pipeline de bout en bout sur les
vraies données du projet.
"""
import pytest

from src.user_itemMatrix import get_matrix_info, get_user_item_matrix


def test_user_item_matrix_has_students_as_rows_and_resources_as_columns():
    matrix = get_user_item_matrix()
    assert matrix.shape[0] > 0
    assert matrix.shape[1] > 0
    assert matrix.index.name == "student_id"


def test_user_item_matrix_scores_are_non_negative():
    matrix = get_user_item_matrix()
    assert (matrix.values >= 0).all()


def test_get_matrix_info_keys_and_consistency():
    info = get_matrix_info()
    for key in ("shape", "n_students", "n_resources", "density", "sparsity"):
        assert key in info

    assert info["density"] + info["sparsity"] == pytest.approx(1.0)
