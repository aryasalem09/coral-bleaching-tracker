from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from backend.config import (
    NOAA_DOWNLOAD_RETRIES,
    NOAA_DOWNLOAD_TIMEOUT_SECONDS,
    NOAA_DOWNLOAD_WORKERS,
)
from backend.noaa_index import clear_noaa_weekly_index_cache
from backend.noaa_products import build_product_path, build_product_url, parse_iso_date, product_kinds

logger = logging.getLogger(__name__)

FILE_LOCKS: dict[Path, threading.Lock] = {}
FILE_LOCKS_GUARD = threading.Lock()


def _file_non_empty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _get_file_lock(path: Path) -> threading.Lock:
    with FILE_LOCKS_GUARD:
        existing = FILE_LOCKS.get(path)
        if existing is not None:
            return existing
        created = threading.Lock()
        FILE_LOCKS[path] = created
        return created


def _download_product_for_date(
    kind: str,
    iso_date: str,
    *,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    output_path = build_product_path(kind, parse_iso_date(iso_date))
    file_lock = _get_file_lock(output_path)

    with file_lock:
        if _file_non_empty(output_path):
            return {"ok": True, "skipped": True, "path": str(output_path), "status_code": 200}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = build_product_url(kind, parse_iso_date(iso_date))
        attempts = max(1, retries + 1)
        last_error = ""
        session = requests.Session()
        session.headers.update({"User-Agent": "coral-bleaching-tracker-noaa-cache/1.0"})

        try:
            for attempt in range(1, attempts + 1):
                temp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with session.get(url, stream=True, timeout=timeout_seconds) as response:
                        status_code = int(response.status_code)
                        if status_code == 404:
                            return {
                                "ok": False,
                                "skipped": False,
                                "path": str(output_path),
                                "status_code": status_code,
                                "error": f"Remote NOAA file was not available for {iso_date}.",
                            }
                        response.raise_for_status()

                        bytes_written = 0
                        with open(temp_path, "wb") as temp_file:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if not chunk:
                                    continue
                                temp_file.write(chunk)
                                bytes_written += len(chunk)
                        if bytes_written <= 0:
                            raise OSError("Downloaded NOAA file was empty.")

                        os.replace(temp_path, output_path)
                        return {
                            "ok": True,
                            "skipped": False,
                            "path": str(output_path),
                            "status_code": status_code,
                            "size_bytes": bytes_written,
                        }
                except (requests.RequestException, OSError) as exc:
                    last_error = str(exc)
                    if attempt < attempts:
                        time.sleep(min(6.0, 1.5 * attempt))
                finally:
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass
        finally:
            session.close()

        logger.warning("NOAA on-demand download failed for %s %s: %s", kind, iso_date, last_error)
        return {
            "ok": False,
            "skipped": False,
            "path": str(output_path),
            "status_code": 0,
            "error": last_error or "NOAA download failed.",
        }


def ensure_weekly_dates_available(iso_dates: list[str]) -> dict[str, Any]:
    normalized_dates = sorted({str(iso_date) for iso_date in iso_dates if iso_date})
    if not normalized_dates:
        return {
            "requested_dates": [],
            "paired_ready_dates": [],
            "downloaded_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "failed_dates": [],
        }

    results_by_date: dict[str, dict[str, dict[str, Any]]] = {iso_date: {} for iso_date in normalized_dates}
    future_to_request: dict[Any, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, NOAA_DOWNLOAD_WORKERS)) as executor:
        for iso_date in normalized_dates:
            for kind in product_kinds():
                future = executor.submit(
                    _download_product_for_date,
                    kind,
                    iso_date,
                    timeout_seconds=NOAA_DOWNLOAD_TIMEOUT_SECONDS,
                    retries=NOAA_DOWNLOAD_RETRIES,
                )
                future_to_request[future] = (iso_date, kind)

        for future in as_completed(future_to_request):
            iso_date, kind = future_to_request[future]
            results_by_date[iso_date][kind] = future.result()

    paired_ready_dates = [
        iso_date
        for iso_date, product_results in results_by_date.items()
        if all(product_results.get(kind, {}).get("ok") for kind in product_kinds())
    ]
    downloaded_files = sum(
        1
        for product_results in results_by_date.values()
        for result in product_results.values()
        if result.get("ok") and not result.get("skipped")
    )
    skipped_files = sum(
        1
        for product_results in results_by_date.values()
        for result in product_results.values()
        if result.get("ok") and result.get("skipped")
    )
    failed_dates = sorted(set(normalized_dates) - set(paired_ready_dates))
    failed_files = sum(
        1
        for product_results in results_by_date.values()
        for result in product_results.values()
        if not result.get("ok")
    )
    if downloaded_files or skipped_files:
        clear_noaa_weekly_index_cache()

    return {
        "requested_dates": normalized_dates,
        "paired_ready_dates": paired_ready_dates,
        "downloaded_files": downloaded_files,
        "skipped_files": skipped_files,
        "failed_files": failed_files,
        "failed_dates": failed_dates,
    }
