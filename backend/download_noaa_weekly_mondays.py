from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests

from backend.config import NOAA_WEEKLY_MANIFEST_PATH, REPO_DIR, ensure_directories
from backend.noaa_index import clear_noaa_weekly_index_cache
from backend.noaa_products import (
    build_product_path,
    build_product_url,
    enumerate_mondays,
    get_product_spec,
    normalize_iso_date,
    parse_iso_date,
    product_kinds,
)

YEAR_LINK_RE = re.compile(r'href="(\d{4})/"')

FILE_LOCKS: dict[Path, threading.Lock] = {}
FILE_LOCKS_GUARD = threading.Lock()
THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class DownloadResult:
    kind: str
    iso_date: str
    ok: bool
    skipped: bool
    size_bytes: int
    status_code: int
    path: Path
    error: str = ""
    remote_listed: bool = True


@dataclass(frozen=True)
class RemoteCoverage:
    kind: str
    years: tuple[int, ...]
    available_dates: frozenset[str]
    first_available_date: str | None
    last_available_date: str | None
    available_monday_count: int
    available_monday_by_year: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NOAA Coral Reef Watch Monday-only weekly DHW + HotSpot files.",
    )
    parser.add_argument("--start", type=str, default=None, help="Optional lower bound in YYYY-MM-DD.")
    parser.add_argument("--end", type=str, default=None, help="Optional upper bound in YYYY-MM-DD.")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent download workers.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per file after the first attempt.")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="HTTP timeout per request.")
    parser.add_argument("--backoff-seconds", type=float, default=1.5, help="Base retry backoff.")
    parser.add_argument(
        "--user-agent",
        type=str,
        default="coral-bleaching-tracker-weekly-noaa-downloader/1.0",
        help="HTTP user agent string.",
    )
    return parser.parse_args()


def get_session(user_agent: str) -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
        THREAD_LOCAL.session = session
    return session


def get_file_lock(path: Path) -> threading.Lock:
    with FILE_LOCKS_GUARD:
        existing = FILE_LOCKS.get(path)
        if existing is not None:
            return existing
        created = threading.Lock()
        FILE_LOCKS[path] = created
        return created


def relative_repo_path(path: Path) -> str:
    return path.relative_to(REPO_DIR).as_posix()


def file_size_if_non_empty(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    return int(size) if size > 0 else 0


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _fetch_text(url: str, *, timeout_seconds: float, user_agent: str) -> str:
    session = get_session(user_agent)
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def _discover_remote_years(kind: str, *, timeout_seconds: float, user_agent: str) -> list[int]:
    spec = get_product_spec(kind)
    html = _fetch_text(spec.base_url + "/", timeout_seconds=timeout_seconds, user_agent=user_agent)
    return sorted({int(year) for year in YEAR_LINK_RE.findall(html)})


def _discover_remote_dates_for_year(kind: str, year: int, *, timeout_seconds: float, user_agent: str) -> list[str]:
    spec = get_product_spec(kind)
    html = _fetch_text(f"{spec.base_url}/{year}/", timeout_seconds=timeout_seconds, user_agent=user_agent)
    html_pattern = re.compile(spec.file_pattern.pattern.lstrip("^").rstrip("$"))
    matches = html_pattern.findall(html)
    return sorted({datetime.strptime(raw_date, "%Y%m%d").date().isoformat() for raw_date in matches})


def discover_remote_coverage(kind: str, *, timeout_seconds: float, user_agent: str) -> RemoteCoverage:
    years = _discover_remote_years(kind, timeout_seconds=timeout_seconds, user_agent=user_agent)
    available_dates: set[str] = set()
    monday_counter: Counter[str] = Counter()

    for year in years:
        year_dates = _discover_remote_dates_for_year(kind, year, timeout_seconds=timeout_seconds, user_agent=user_agent)
        available_dates.update(year_dates)
        for iso_date in year_dates:
            if parse_iso_date(iso_date).weekday() == 0:
                monday_counter[str(year)] += 1

    sorted_dates = sorted(available_dates)
    monday_count = sum(monday_counter.values())
    return RemoteCoverage(
        kind=kind,
        years=tuple(years),
        available_dates=frozenset(sorted_dates),
        first_available_date=sorted_dates[0] if sorted_dates else None,
        last_available_date=sorted_dates[-1] if sorted_dates else None,
        available_monday_count=int(monday_count),
        available_monday_by_year=dict(sorted(monday_counter.items())),
    )


def _download_file_with_retries(
    *,
    kind: str,
    iso_date: str,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
    user_agent: str,
    remote_dates: frozenset[str],
) -> DownloadResult:
    day = parse_iso_date(iso_date)
    output_path = build_product_path(kind, day)
    file_lock = get_file_lock(output_path)

    with file_lock:
        existing_size = file_size_if_non_empty(output_path)
        if existing_size > 0:
            return DownloadResult(
                kind=kind,
                iso_date=iso_date,
                ok=True,
                skipped=True,
                size_bytes=existing_size,
                status_code=200,
                path=output_path,
            )

        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

        if iso_date not in remote_dates:
            return DownloadResult(
                kind=kind,
                iso_date=iso_date,
                ok=False,
                skipped=False,
                size_bytes=0,
                status_code=404,
                path=output_path,
                error="Date is not listed in the remote NOAA directory index.",
                remote_listed=False,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        session = get_session(user_agent)
        url = build_product_url(kind, day)
        total_attempts = max(1, retries + 1)
        last_error = "download failed"

        for attempt in range(1, total_attempts + 1):
            temp_path = output_path.with_name(
                f".{output_path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with session.get(url, stream=True, timeout=timeout_seconds) as response:
                    status_code = int(response.status_code)

                    if status_code == 404:
                        return DownloadResult(
                            kind=kind,
                            iso_date=iso_date,
                            ok=False,
                            skipped=False,
                            size_bytes=0,
                            status_code=status_code,
                            path=output_path,
                            error=f"http 404: {url}",
                        )

                    if status_code >= 500:
                        raise requests.HTTPError(f"server returned http {status_code}")

                    if status_code != 200:
                        return DownloadResult(
                            kind=kind,
                            iso_date=iso_date,
                            ok=False,
                            skipped=False,
                            size_bytes=0,
                            status_code=status_code,
                            path=output_path,
                            error=f"http {status_code}: {url}",
                        )

                    bytes_written = 0
                    with open(temp_path, "wb") as temp_file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            temp_file.write(chunk)
                            bytes_written += len(chunk)

                    if bytes_written <= 0:
                        raise OSError("empty download")

                    os.replace(temp_path, output_path)
                    return DownloadResult(
                        kind=kind,
                        iso_date=iso_date,
                        ok=True,
                        skipped=False,
                        size_bytes=bytes_written,
                        status_code=status_code,
                        path=output_path,
                    )
            except (requests.RequestException, OSError) as exc:
                last_error = str(exc)
            finally:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

            if attempt < total_attempts:
                sleep_for = backoff_seconds * (2 ** (attempt - 1))
                sleep_for += random.uniform(0.0, 0.35)
                time.sleep(sleep_for)

        return DownloadResult(
            kind=kind,
            iso_date=iso_date,
            ok=False,
            skipped=False,
            size_bytes=0,
            status_code=0,
            path=output_path,
            error=last_error,
        )


def _requested_monday_dates(
    remote_coverages: dict[str, RemoteCoverage],
    *,
    start: str | None,
    end: str | None,
) -> list[str]:
    first_dates = [coverage.first_available_date for coverage in remote_coverages.values() if coverage.first_available_date]
    last_dates = [coverage.last_available_date for coverage in remote_coverages.values() if coverage.last_available_date]
    if not first_dates or not last_dates:
        return []

    effective_start = max(first_dates)
    effective_end = min(last_dates)
    if start is not None:
        effective_start = max(effective_start, start)
    if end is not None:
        effective_end = min(effective_end, end)

    mondays = enumerate_mondays(parse_iso_date(effective_start), parse_iso_date(effective_end))
    return [day.isoformat() for day in mondays]


def _build_manifest(
    *,
    remote_coverages: dict[str, RemoteCoverage],
    requested_dates: list[str],
    results_by_date: dict[str, dict[str, DownloadResult]],
) -> dict[str, Any]:
    date_status: dict[str, Any] = {}
    ok_dates: list[str] = []
    failed_dates: list[str] = []
    product_success_counts: Counter[str] = Counter()
    product_fail_counts: Counter[str] = Counter()

    for iso_date in requested_dates:
        product_results = results_by_date.get(iso_date, {})
        date_ok = all(product_results.get(kind) and product_results[kind].ok for kind in product_kinds())
        if date_ok:
            ok_dates.append(iso_date)
        else:
            failed_dates.append(iso_date)

        products_payload: dict[str, Any] = {}
        for kind in product_kinds():
            result = product_results.get(kind)
            if result is None:
                product_fail_counts[kind] += 1
                products_payload[kind] = {
                    "ok": False,
                    "status_code": 0,
                    "size_bytes": 0,
                    "path": relative_repo_path(build_product_path(kind, parse_iso_date(iso_date))),
                    "error": "No download result was recorded.",
                    "skipped": False,
                    "remote_listed": False,
                }
                continue
            if result.ok:
                product_success_counts[kind] += 1
            else:
                product_fail_counts[kind] += 1
            products_payload[kind] = {
                "ok": result.ok,
                "status_code": result.status_code,
                "size_bytes": result.size_bytes,
                "path": relative_repo_path(result.path),
                "error": result.error,
                "skipped": result.skipped,
                "remote_listed": result.remote_listed,
            }

        date_status[iso_date] = {
            "ok": date_ok,
            "products": products_payload,
        }

    by_date = {
        iso_date: {
            kind: payload["path"]
            for kind, payload in entry["products"].items()
            if payload["ok"]
        }
        for iso_date, entry in date_status.items()
        if entry["ok"]
    }

    return {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "schedule": "weekly_mondays",
        "requested_dates": requested_dates,
        "ok_dates": ok_dates,
        "failed_dates": failed_dates,
        "product_coverage": {
            kind: {
                "base_url": get_product_spec(kind).base_url,
                "directory": relative_repo_path(get_product_spec(kind).directory),
                "available_years": list(coverage.years),
                "remote_available_date_count": len(coverage.available_dates),
                "remote_available_monday_count": coverage.available_monday_count,
                "remote_available_monday_by_year": coverage.available_monday_by_year,
                "first_available_date": coverage.first_available_date,
                "last_available_date": coverage.last_available_date,
                "local_success_count": int(product_success_counts[kind]),
                "local_failure_count": int(product_fail_counts[kind]),
            }
            for kind, coverage in remote_coverages.items()
        },
        "first_available_dates": {
            kind: coverage.first_available_date
            for kind, coverage in remote_coverages.items()
        },
        "last_available_dates": {
            kind: coverage.last_available_date
            for kind, coverage in remote_coverages.items()
        },
        "date_status": date_status,
        "by_date": by_date,
    }


def download_weekly_mondays(args: argparse.Namespace | None = None) -> dict[str, Any]:
    options = args or parse_args()
    ensure_directories()

    start = normalize_iso_date(options.start)
    end = normalize_iso_date(options.end)
    remote_coverages = {
        kind: discover_remote_coverage(kind, timeout_seconds=options.timeout_seconds, user_agent=options.user_agent)
        for kind in product_kinds()
    }
    requested_dates = _requested_monday_dates(remote_coverages, start=start, end=end)
    results_by_date: dict[str, dict[str, DownloadResult]] = {iso_date: {} for iso_date in requested_dates}

    futures = []
    with ThreadPoolExecutor(max_workers=max(1, int(options.workers))) as executor:
        for iso_date in requested_dates:
            for kind in product_kinds():
                futures.append(
                    executor.submit(
                        _download_file_with_retries,
                        kind=kind,
                        iso_date=iso_date,
                        timeout_seconds=float(options.timeout_seconds),
                        retries=int(options.retries),
                        backoff_seconds=float(options.backoff_seconds),
                        user_agent=str(options.user_agent),
                        remote_dates=remote_coverages[kind].available_dates,
                    )
                )

        for future in as_completed(futures):
            result = future.result()
            results_by_date[result.iso_date][result.kind] = result

    manifest = _build_manifest(
        remote_coverages=remote_coverages,
        requested_dates=requested_dates,
        results_by_date=results_by_date,
    )
    _atomic_write_json(NOAA_WEEKLY_MANIFEST_PATH, manifest)
    clear_noaa_weekly_index_cache()

    ok_dates = manifest["ok_dates"]
    failed_dates = manifest["failed_dates"]
    print(
        f"Downloaded weekly NOAA Mondays: {len(ok_dates)} succeeded, {len(failed_dates)} failed, "
        f"manifest written to {NOAA_WEEKLY_MANIFEST_PATH}"
    )
    return manifest


if __name__ == "__main__":
    download_weekly_mondays()
