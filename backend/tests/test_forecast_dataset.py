from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from backend.ml.build_dataset import attach_future_bleaching_targets
from backend.ml.feature_definitions import TARGET_COLUMN
from backend.ml.split_strategy import assign_time_split
from backend.noaa import get_site_weekly_feature_context


class ForecastDatasetTests(unittest.TestCase):
    def test_future_labels_use_only_post_anchor_observations(self) -> None:
        feature_rows = pd.DataFrame(
            [
                {"site_id": "A", "date": "2020-01-06"},
                {"site_id": "B", "date": "2020-01-06"},
                {"site_id": "C", "date": "2020-01-06"},
            ]
        )
        observations = pd.DataFrame(
            [
                {"site_id": "A", "date": "2020-01-06", "observed_percent_bleaching": 75.0},
                {"site_id": "A", "date": "2020-01-10", "observed_percent_bleaching": 0.0},
                {"site_id": "A", "date": "2020-01-25", "observed_percent_bleaching": 20.0},
                {"site_id": "B", "date": "2020-01-20", "observed_percent_bleaching": 0.0},
                {"site_id": "C", "date": "2020-03-10", "observed_percent_bleaching": 60.0},
            ]
        )

        result = attach_future_bleaching_targets(feature_rows, observations)
        rows = result.set_index("site_id")

        self.assertTrue(bool(rows.loc["A", "target_eligible"]))
        self.assertEqual(int(rows.loc["A", TARGET_COLUMN]), 1)
        self.assertEqual(int(rows.loc["A", "future_observation_count_4w"]), 2)
        self.assertEqual(int(rows.loc["A", "future_positive_observation_count_4w"]), 1)
        self.assertEqual(str(pd.to_datetime(rows.loc["A", "first_future_observation_date"]).date()), "2020-01-10")
        self.assertEqual(int(rows.loc["A", "days_to_first_future_observation"]), 4)

        self.assertTrue(bool(rows.loc["B", "target_eligible"]))
        self.assertEqual(int(rows.loc["B", TARGET_COLUMN]), 0)
        self.assertEqual(int(rows.loc["B", "future_positive_observation_count_4w"]), 0)

        self.assertFalse(bool(rows.loc["C", "target_eligible"]))
        self.assertTrue(pd.isna(rows.loc["C", TARGET_COLUMN]))

    def test_time_split_purges_rows_near_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2012-12-03",
                        "2012-12-10",
                        "2013-01-07",
                        "2016-12-05",
                        "2017-01-02",
                    ]
                )
            }
        )

        split = assign_time_split(frame)
        self.assertEqual(split["split"].tolist(), ["train", "excluded", "validation", "excluded", "test"])

    @mock.patch(
        "backend.noaa.sample_site_environmental_context",
        side_effect=lambda lat, lon, iso_date, index=None: {
            "hotspot": 1.25,
            "dhw": 2.75,
            "used_lat": float(lat),
            "used_lon": float(lon),
            "snap_km": 0.0,
            "snapped": False,
        },
    )
    def test_live_feature_context_stays_on_monday_issue_date(self, _mock_sample: mock.Mock) -> None:
        class FakeAvailability:
            def nearest_previous_monday(self, iso_date: str) -> str:
                return "2020-03-09"

            def all_available_monday_dates(self) -> list[str]:
                return self.recent_history_dates("2020-03-09", weeks=12)

            def recent_history_dates(self, iso_date: str, weeks: int) -> list[str]:
                return pd.date_range(end=iso_date, periods=weeks, freq="W-MON").strftime("%Y-%m-%d").tolist()

        result = get_site_weekly_feature_context(
            lat=-17.53,
            lon=177.12,
            requested_date="2020-03-12",
            index=FakeAvailability(),
        )

        self.assertEqual(result["requested_date"], "2020-03-12")
        self.assertEqual(result["used_date"], "2020-03-09")
        self.assertEqual(result["weekly_anchor_date"], "2020-03-09")
        self.assertEqual(result["days_since_anchor_monday"], 0)
        self.assertEqual(result["weekly_history_weeks_available"], 12)
        self.assertEqual(len(result["history_records"]), 12)


if __name__ == "__main__":
    unittest.main()
