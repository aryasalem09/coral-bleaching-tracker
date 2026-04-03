from __future__ import annotations

from datetime import date
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from backend.api import app


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.sample_site_id = "3579"
        cls.sample_date = "2020-03-12"

    def test_model_status_endpoint_reports_runtime_metadata(self) -> None:
        response = self.client.get("/api/model/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("model_loaded", payload)
        self.assertIn("model_version", payload)
        self.assertIn("artifact_path", payload)
        self.assertIn("sklearn_version", payload)
        self.assertIn("loader_error", payload)
        self.assertTrue(payload["model_loaded"])

    def test_predict_endpoint_success_shape(self) -> None:
        response = self.client.post(
            "/api/predict",
            json={"site_id": self.sample_site_id, "date": self.sample_date, "prefer_live": False},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["status"], "available")
        self.assertIsInstance(payload["probability"], float)
        self.assertIsInstance(payload["predicted_event"], bool)
        self.assertIsInstance(payload["threshold"], float)
        self.assertEqual(payload["prediction_unit"], "site-anchor-date")
        self.assertEqual(payload["forecast_horizon_weeks"], 4)
        self.assertIn("next 4 weeks", payload["probability_meaning"])
        self.assertIn(payload["context_source"], {"historical_forecast_row", "weekly_noaa_history"})
        self.assertIn("feature_date_used", payload)
        self.assertIn("coverage_notes", payload)
        self.assertLessEqual(
            date.fromisoformat(payload["forecast_issue_date"]),
            date.fromisoformat(self.sample_date),
        )
        self.assertEqual(payload["weekly_anchor_date"], payload["forecast_issue_date"])

    @mock.patch(
        "backend.api.get_model_runtime_status",
        return_value={
            "status": "invalid",
            "ready": False,
            "model_loaded": False,
            "message": "Forecast model unavailable in the current backend environment.",
            "artifact_path": "backend/ml/artifacts/bleaching_event_model.joblib",
            "model_version": "2026.04.01",
            "sklearn_version": "1.6.1",
            "trained_with_sklearn_version": "1.6.1",
            "loader_error": "boom",
        },
    )
    def test_predict_endpoint_unavailable_when_model_not_loaded(self, _mock_runtime: mock.Mock) -> None:
        response = self.client.post(
            "/api/predict",
            json={"site_id": self.sample_site_id, "date": self.sample_date, "prefer_live": False},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["status"], "model_unavailable")
        self.assertIn("Forecast unavailable", payload["message"])
        self.assertNotIn("probability", payload)

    @mock.patch(
        "backend.api._build_weekly_history_payload",
        return_value={
            "available": False,
            "requested_date": "2020-03-12",
            "message": "Weekly NOAA history unavailable in this test.",
            "records": [],
        },
    )
    def test_selected_site_analysis_schema_distinguishes_observed_from_weekly_history(
        self,
        _mock_weekly_history: mock.Mock,
    ) -> None:
        response = self.client.get(f"/api/site/{self.sample_site_id}/analysis", params={"date": self.sample_date, "prefer_live": False})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["selected_observed_date"], self.sample_date)
        self.assertIn("observed_summary", payload)
        self.assertIn("observed_timeline", payload)
        self.assertIn("environmental_noaa", payload)
        self.assertIn("prediction", payload)
        self.assertIn("model_metadata", payload)
        self.assertIn("data_availability", payload)
        self.assertGreater(len(payload["observed_timeline"]["records"]), 0)
        self.assertIn("observation_sparsity_note", payload["observed_summary"])
        self.assertFalse(payload["environmental_noaa"]["weekly_history"]["available"])
        self.assertIsInstance(payload["environmental_noaa"]["weekly_history"]["records"], list)


if __name__ == "__main__":
    unittest.main()
