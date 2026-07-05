import pandas as pd
import pytest

from src import CF


@pytest.fixture
def tiny_matrix():
    """Petite matrice utilisateur-ressource en mémoire (5 étudiants x
    6 ressources), assez grande pour un KNN avec peu de voisins."""
    import numpy as np

    rng = np.random.RandomState(0)
    data = rng.randint(0, 5, size=(5, 6)).astype(float)
    return pd.DataFrame(
        data,
        index=[f"STU-2026-{i:04d}" for i in range(1, 6)],
        columns=[f"RES-{i:03d}" for i in range(1, 7)],
    )


@pytest.fixture(autouse=True)
def small_cf_config(monkeypatch, tiny_matrix):
    """Recalibre les constantes de CF.py (importées depuis cf_config au
    chargement du module) pour qu'elles soient cohérentes avec la petite
    matrice de test, et remplace get_user_item_matrix par la fixture en
    mémoire — évite toute dépendance au dataset réel du projet."""
    monkeypatch.setattr(CF, "get_user_item_matrix", lambda: tiny_matrix)
    monkeypatch.setattr(CF, "SVD_TEST_K", 2)
    monkeypatch.setattr(CF, "SVD_COMPONENTS", 3)
    monkeypatch.setattr(CF, "SVD_VARIANCE_THRESHOLD", 0.99)  # force raw_knn par défaut
    monkeypatch.setattr(CF, "KNN_NEIGHBORS", 2)
    monkeypatch.setattr(CF, "MAX_NEIGHBORS", 4)
    yield


def test_build_cf_model_raw_knn_when_variance_below_threshold(tmp_path):
    save_path = tmp_path / "cf_model.pkl"
    model_data = CF.build_cf_model(save_path=str(save_path))

    assert model_data["type"] == "raw"
    assert save_path.exists()


def test_build_cf_model_svd_when_variance_above_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(CF, "SVD_VARIANCE_THRESHOLD", 0.0)  # toujours satisfait
    save_path = tmp_path / "cf_model.pkl"
    model_data = CF.build_cf_model(save_path=str(save_path))

    assert model_data["type"] == "svd"
    assert "U_reduced" in model_data


def test_load_cf_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CF.load_cf_model(str(tmp_path / "nope.pkl"))


def test_load_cf_model_roundtrip(tmp_path):
    save_path = tmp_path / "cf_model.pkl"
    CF.build_cf_model(save_path=str(save_path))
    loaded = CF.load_cf_model(str(save_path))
    assert "matrix" in loaded and "knn" in loaded


def test_get_or_build_cf_model_builds_when_absent(tmp_path):
    save_path = tmp_path / "cf_model.pkl"
    assert not save_path.exists()
    CF.get_or_build_cf_model(str(save_path))
    assert save_path.exists()


def test_get_or_build_cf_model_reuses_valid_cache(tmp_path, monkeypatch):
    save_path = tmp_path / "cf_model.pkl"
    CF.get_or_build_cf_model(str(save_path))
    mtime_1 = save_path.stat().st_mtime_ns

    monkeypatch.setattr(CF, "is_cache_valid", lambda model_data, files: True)
    CF.get_or_build_cf_model(str(save_path))
    mtime_2 = save_path.stat().st_mtime_ns

    assert mtime_1 == mtime_2


def test_get_or_build_cf_model_rebuilds_when_invalid(tmp_path, monkeypatch):
    save_path = tmp_path / "cf_model.pkl"
    CF.get_or_build_cf_model(str(save_path))
    mtime_1 = save_path.stat().st_mtime_ns

    monkeypatch.setattr(CF, "is_cache_valid", lambda model_data, files: False)
    CF.get_or_build_cf_model(str(save_path))
    mtime_2 = save_path.stat().st_mtime_ns

    assert mtime_2 != mtime_1


# ── CFEngine ────────────────────────────────────────────────────────


def test_cf_engine_recommends_for_existing_student(tmp_path):
    engine = CF.CFEngine(model_path=str(tmp_path / "cf_model.pkl"))
    result = engine.get_recommendations("STU-2026-0001", n_recommendations=2)
    assert len(result["recommended_resources"]) <= 2


def test_cf_engine_unknown_student_returns_empty_result(tmp_path):
    engine = CF.CFEngine(model_path=str(tmp_path / "cf_model.pkl"))
    result = engine.get_recommendations("GHOST-STUDENT")

    assert result["student_exists"] is False
    assert result["recommended_resources"] == []
    assert result["num_similar_students"] == 0


def test_cf_engine_svd_path_returns_recommendations(tmp_path, monkeypatch):
    monkeypatch.setattr(CF, "SVD_VARIANCE_THRESHOLD", 0.0)
    engine = CF.CFEngine(model_path=str(tmp_path / "cf_model.pkl"))
    result = engine.get_recommendations("STU-2026-0002", n_recommendations=2)

    assert result["method"] == "svd_knn"
    assert engine.use_svd is True


def test_cf_engine_get_status_contains_expected_keys(tmp_path):
    engine = CF.CFEngine(model_path=str(tmp_path / "cf_model.pkl"))
    status = engine.get_status()

    for key in (
        "matrix_shape",
        "svd_enabled",
        "svd_variance_at_k5",
        "svd_variance_threshold",
        "active_method",
        "knn_neighbors",
    ):
        assert key in status


def test_cf_engine_already_rated_resources_are_excluded(tmp_path, tiny_matrix):
    engine = CF.CFEngine(model_path=str(tmp_path / "cf_model.pkl"))
    result = engine.get_recommendations("STU-2026-0001", n_recommendations=6)

    already_rated = set(
        tiny_matrix.columns[tiny_matrix.loc["STU-2026-0001"] > 0]
    )
    recommended_ids = {r["resource_id"] for r in result["recommended_resources"]}
    assert already_rated.isdisjoint(recommended_ids)


def test_module_level_get_recommendations_uses_singleton(tmp_path, monkeypatch):
    CF.reset_cf_engine()
    new_path = str(tmp_path / "cf_model.pkl")
    monkeypatch.setattr(CF, "CF_MODEL_PATH", new_path)
    # `CF.CF_MODEL_PATH` seul ne suffit pas : CFEngine.__init__(self, model_path=CF_MODEL_PATH)
    # a figé cette valeur par défaut au moment de la définition de la classe,
    # pas à chaque appel. Il faut aussi patcher le défaut lié à la fonction.
    monkeypatch.setattr(CF.CFEngine.__init__, "__defaults__", (new_path,))
    result = CF.get_recommendations("STU-2026-0001", n_recommendations=1)
    assert result["student_exists"] is True
    assert CF.get_cf_engine() is CF.get_cf_engine()  # même instance réutilisée
    CF.reset_cf_engine()


def test_reset_cf_engine_forces_new_instance(tmp_path, monkeypatch):
    new_path = str(tmp_path / "cf_model.pkl")
    monkeypatch.setattr(CF, "CF_MODEL_PATH", new_path)
    monkeypatch.setattr(CF.CFEngine.__init__, "__defaults__", (new_path,))
    CF.reset_cf_engine()
    engine_1 = CF.get_cf_engine()
    CF.reset_cf_engine()
    engine_2 = CF.get_cf_engine()
    assert engine_1 is not engine_2
    CF.reset_cf_engine()


@pytest.fixture
def undersized_matrix():
    """3 étudiants x 8 ressources : avec n_components=6 demandé, TruncatedSVD
    ne peut renvoyer que 3 composantes réelles (bornées par n_samples)."""
    import numpy as np

    rng = np.random.RandomState(0)
    data = rng.randint(0, 5, size=(3, 8)).astype(float)
    return pd.DataFrame(
        data,
        index=[f"STU-2026-{i:04d}" for i in range(1, 4)],
        columns=[f"RES-{i:03d}" for i in range(1, 9)],
    )


def test_cf_handles_fewer_components_gracefully(
    tmp_path, monkeypatch, undersized_matrix
):
    monkeypatch.setattr(CF, "get_user_item_matrix", lambda: undersized_matrix)
    monkeypatch.setattr(CF, "SVD_TEST_K", 2)
    monkeypatch.setattr(CF, "SVD_COMPONENTS", 6)
    monkeypatch.setattr(CF, "SVD_VARIANCE_THRESHOLD", 0.0)
    monkeypatch.setattr(CF, "KNN_NEIGHBORS", 2)
    monkeypatch.setattr(CF, "MAX_NEIGHBORS", 2)

    engine = CF.CFEngine(model_path=str(tmp_path / "cf_model.pkl"))
    result = engine.get_recommendations("STU-2026-0001", n_recommendations=2)
    assert result["student_exists"] is True
