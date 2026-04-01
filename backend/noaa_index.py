from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.config import NOAA_MANIFEST_PATH, NOAA_WEEKLY_MANIFEST_PATH, REPO_DIR
from backend.noaa_products import (
    build_product_path,
    get_product_spec,
    normalize_iso_date,
    parse_iso_date,
    parse_product_date_from_filename,
    product_kinds,
)


def _file_non_empty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _resolve_manifest_file_path(path_value: Any) -> Path | None:
    if path_value is None:
        return None
    raw_text = str(path_value).strip()
    if not raw_text:
        return None
    raw_path = Path(raw_text)
    if raw_path.is_absolute():
        return raw_path
    return (REPO_DIR / raw_path).resolve()


def _scan_local_product_files(kind: str) -> dict[str, Path]:
    spec = get_product_spec(kind)
    found: dict[str, Path] = {}
    if not spec.directory.exists():
        return found
    for file_path in spec.directory.iterdir():
        iso_date = parse_product_date_from_filename(kind, file_path.name)
        if iso_date is None or not _file_non_empty(file_path):
            continue
        found[iso_date] = file_path
    return found


def _scan_local_index() -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, Path]]]:
    product_paths = {kind: _scan_local_product_files(kind) for kind in product_kinds()}
    paired_dates = sorted(set.intersection(*(set(paths.keys()) for paths in product_paths.values()))) if product_paths else []
    paired_paths = {
        iso_date: {kind: product_paths[kind][iso_date] for kind in product_kinds()}
        for iso_date in paired_dates
    }
    product_dates = {
        kind: tuple(sorted(paths.keys()))
        for kind, paths in product_paths.items()
    }
    return product_dates, paired_paths


def _coerce_manifest_product_entry(kind: str, iso_date: str, entry: Any) -> Path | None:
    default_path = build_product_path(kind, parse_iso_date(iso_date))
    if isinstance(entry, dict):
        path_candidate = (
            entry.get("path")
            or entry.get("file")
            or entry.get("relative_path")
            or entry.get(kind)
        )
    else:
        path_candidate = entry
    resolved = _resolve_manifest_file_path(path_candidate)
    return resolved or default_path


def _load_manifest_index() -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, Path]], str] | None:
    manifest_path = NOAA_WEEKLY_MANIFEST_PATH if NOAA_WEEKLY_MANIFEST_PATH.exists() else NOAA_MANIFEST_PATH
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    product_paths: dict[str, dict[str, Path]] = {kind: {} for kind in product_kinds()}

    date_records = manifest.get("date_status") if isinstance(manifest, dict) else None
    if isinstance(date_records, dict):
        for raw_date, status in date_records.items():
            iso_date = normalize_iso_date(raw_date)
            if iso_date is None or not isinstance(status, dict):
                continue
            product_status = status.get("products", {})
            for kind in product_kinds():
                entry = product_status.get(kind)
                if not isinstance(entry, dict) or not entry.get("ok"):
                    continue
                file_path = _coerce_manifest_product_entry(kind, iso_date, entry)
                if file_path is not None and _file_non_empty(file_path):
                    product_paths[kind][iso_date] = file_path
    else:
        ok_dates = manifest.get("ok_dates", [])
        by_date = manifest.get("by_date", {})
        for raw_date in ok_dates if isinstance(ok_dates, list) else []:
            iso_date = normalize_iso_date(raw_date)
            if iso_date is None:
                continue
            entry = by_date.get(iso_date, {}) if isinstance(by_date, dict) else {}
            for kind in product_kinds():
                file_path = _coerce_manifest_product_entry(kind, iso_date, entry.get(kind) if isinstance(entry, dict) else None)
                if file_path is not None and _file_non_empty(file_path):
                    product_paths[kind][iso_date] = file_path

    paired_dates = sorted(set.intersection(*(set(paths.keys()) for paths in product_paths.values()))) if product_paths else []
    if not paired_dates:
        return None

    paired_paths = {
        iso_date: {kind: product_paths[kind][iso_date] for kind in product_kinds()}
        for iso_date in paired_dates
    }
    product_dates = {kind: tuple(sorted(paths.keys())) for kind, paths in product_paths.items()}
    source = "weekly_manifest" if manifest_path == NOAA_WEEKLY_MANIFEST_PATH else "legacy_manifest"
    return product_dates, paired_paths, source


@dataclass(frozen=True)
class NoaaAvailabilityIndex:
    product_dates: dict[str, tuple[str, ...]]
    paired_paths: dict[str, dict[str, Path]]
    source: str

    def paired_dates(self) -> tuple[str, ...]:
        return tuple(sorted(self.paired_paths.keys()))

    def all_available_monday_dates(self) -> list[str]:
        return list(self.paired_dates())

    def available_monday_dates_for_product(self, kind: str) -> list[str]:
        return list(self.product_dates.get(kind, ()))

    def get_paths_for_date(self, iso_date: str) -> dict[str, Path] | None:
        normalized = normalize_iso_date(iso_date)
        if normalized is None:
            return None
        return self.paired_paths.get(normalized)

    def nearest_previous_monday(self, iso_date: str | date, *, require_pair: bool = True) -> str | None:
        date_list = self.paired_dates() if require_pair else tuple(sorted(self.product_dates.get("dhw", ())))
        if not date_list:
            return None
        candidate = iso_date.isoformat() if isinstance(iso_date, date) else normalize_iso_date(iso_date)
        if candidate is None:
            return None
        index = bisect_right(date_list, candidate) - 1
        if index < 0:
            return None
        return date_list[index]

    def recent_history_dates(self, iso_date: str | date, *, weeks: int, require_pair: bool = True) -> list[str]:
        if weeks <= 0:
            return []
        date_list = self.paired_dates() if require_pair else tuple(sorted(self.product_dates.get("dhw", ())))
        if not date_list:
            return []
        candidate = iso_date.isoformat() if isinstance(iso_date, date) else normalize_iso_date(iso_date)
        if candidate is None:
            return []
        end_index = bisect_right(date_list, candidate)
        start_index = max(0, end_index - weeks)
        return list(date_list[start_index:end_index])

    def coverage_summary(self) -> dict[str, Any]:
        paired_dates = self.paired_dates()
        return {
            "source": self.source,
            "paired_date_count": len(paired_dates),
            "paired_first_date": paired_dates[0] if paired_dates else None,
            "paired_last_date": paired_dates[-1] if paired_dates else None,
            "products": {
                kind: {
                    "date_count": len(self.product_dates.get(kind, ())),
                    "first_date": self.product_dates.get(kind, (None,))[0] if self.product_dates.get(kind) else None,
                    "last_date": self.product_dates.get(kind, (None,))[-1] if self.product_dates.get(kind) else None,
                    "directory": str(get_product_spec(kind).directory),
                }
                for kind in product_kinds()
            },
        }


@lru_cache(maxsize=1)
def get_noaa_weekly_index() -> NoaaAvailabilityIndex:
    manifest_index = _load_manifest_index()
    scanned_product_dates, scanned_paired_paths = _scan_local_index()

    if manifest_index is None:
        return NoaaAvailabilityIndex(product_dates=scanned_product_dates, paired_paths=scanned_paired_paths, source="scan")

    manifest_product_dates, manifest_paired_paths, manifest_source = manifest_index
    merged_product_dates = {
        kind: tuple(
            sorted(
                set(scanned_product_dates.get(kind, ())) | set(manifest_product_dates.get(kind, ()))
            )
        )
        for kind in product_kinds()
    }
    merged_paired_paths = {**manifest_paired_paths, **scanned_paired_paths}
    merged_source = manifest_source if not scanned_paired_paths else f"{manifest_source}+scan"
    return NoaaAvailabilityIndex(product_dates=merged_product_dates, paired_paths=merged_paired_paths, source=merged_source)


def clear_noaa_weekly_index_cache() -> None:
    get_noaa_weekly_index.cache_clear()
