from fraudservice.drift import check, psi
from tests.conftest import make_data


def test_psi_is_zero_for_identical_distribution():
    assert psi([0.5, 0.5], [0.0, 1.0], [-1, 0.5, 2]) < 1e-9


def test_no_drift_on_similar_data(model_dir):
    report = check(make_data(5000, seed=2), model_dir)
    assert report["drifted_features"] == []
    assert report["needs_attention"] is False


def test_detects_shifted_data(model_dir):
    report = check(make_data(5000, shift=1.5, seed=3), model_dir)
    assert len(report["drifted_features"]) > 0
    assert report["needs_attention"] is True


def test_reports_performance_when_labels_present(model_dir):
    report = check(make_data(5000, seed=4), model_dir)
    assert "performance" in report
    assert 0 <= report["performance"]["recall"] <= 1
