"""The app, the evaluation and pre-resolve use the synthetic cases plus Ryde's
sample cases; the queue marks Ryde's cases so the UI can list them first."""

import pytest
from fastapi.testclient import TestClient

from ridecourt.data import load_app_dataset, load_dataset
from ridecourt.ryde_adapter import RYDE_SAMPLES_DIR

pytestmark = pytest.mark.skipif(
    not RYDE_SAMPLES_DIR.exists(), reason="data/ryde_samples/ directory not found"
)


def test_app_dataset_adds_ryde_cases_with_labels():
    base = load_dataset()
    ds = load_app_dataset()
    assert ds.sample_ids == ["DISP-001", "DISP-002", "DISP-003"]
    assert len(ds.disputes) == len(base.disputes) + 3
    for did in ds.sample_ids:
        assert ds.label(did) is not None
        dispute = ds.dispute(did)
        assert ds.trip(dispute.trip_id) is not None


def test_queue_marks_ryde_samples(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("STATE_BACKEND", "memory")
    from ridecourt.api import create_app

    client = TestClient(create_app())
    items = client.get("/api/disputes").json()["disputes"]
    flagged = sorted(d["dispute_id"] for d in items if d["ryde_sample"])
    assert flagged == ["DISP-001", "DISP-002", "DISP-003"]
