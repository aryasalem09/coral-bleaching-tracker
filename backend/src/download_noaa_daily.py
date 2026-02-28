import argparse
import json
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

DHW_BASE_URL = (
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/"
    "5km/v3.1_op/nc/v1.0/daily/dhw"
)
HS_BASE_URL = (
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/"
    "5km/v3.1_op/nc/v1.0/daily/hs"
)

DHW_TEMPLATE = "ct5km_dhw_v3.1_{ymd}.nc"
HS_TEMPLATE = "ct5km_hs_v3.1_{ymd}.nc"

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
RAW_DIR = BACKEND_DIR / "data" / "raw"
DHW_DIR = RAW_DIR / "noaa_dhw"
HS_DIR = RAW_DIR / "noaa_hs"
MANIFEST_PATH = RAW_DIR / "noaa_manifest.json"

FILE_LOCKS: dict[Path, threading.Lock] = {}
FILE_LOCKS_GUARD = threading.Lock()
THREAD_LOCAL = threading.local()


@dataclass
class DownloadResult:
    ok: bool
    path: Path
    size: int
    status: int
    skipped: bool = False
    error: str = ""


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date '{value}', expected YYYY-MM-DD") from exc


def date_range_inclusive(start_date: date, end_date: date) -> list[date]:
    total_days = (end_date - start_date).days
    return [start_date + timedelta(days=offset) for offset in range(total_days + 1)]


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


def file_size_if_non_empty(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    return size if size > 0 else 0


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def relative_repo_path(path: Path) -> str:
    return path.relative_to(REPO_DIR).as_posix()


def build_filename(kind: str, day: date) -> str:
    ymd = day.strftime("%Y%m%d")
    if kind == "dhw":
        return DHW_TEMPLATE.format(ymd=ymd)
    if kind == "hs":
        return HS_TEMPLATE.format(ymd=ymd)
    raise ValueError(f"unknown kind: {kind}")


def build_url(kind: str, day: date) -> str:
    year = day.strftime("%Y")
    filename = build_filename(kind, day)
    if kind == "dhw":
        return f"{DHW_BASE_URL}/{year}/{filename}"
    if kind == "hs":
        return f"{HS_BASE_URL}/{year}/{filename}"
    raise ValueError(f"unknown kind: {kind}")


def download_file_with_retries(
    url: str,
    output_path: Path,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
    user_agent: str,
) -> DownloadResult:
    ensure_parent_dir(output_path)
    file_lock = get_file_lock(output_path)

    with file_lock:
        existing_size = file_size_if_non_empty(output_path)
        if existing_size > 0:
            return DownloadResult(ok=True, path=output_path, size=existing_size, status=200, skipped=True)

        if output_path.exists() and existing_size == 0:
            try:
                output_path.unlink()
            except OSError:
                pass

        session = get_session(user_agent)
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
                            ok=False,
                            path=output_path,
                            size=0,
                            status=status_code,
                            error=f"http 404: {url}",
                        )

                    if status_code >= 500:
                        raise requests.HTTPError(f"server returned http {status_code}")

                    if status_code != 200:
                        return DownloadResult(
                            ok=False,
                            path=output_path,
                            size=0,
                            status=status_code,
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
                        ok=True,
                        path=output_path,
                        size=bytes_written,
                        status=status_code,
                        skipped=False,
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
                sleep_for += random.uniform(0.0, 0.25)
                time.sleep(sleep_for)

        return DownloadResult(ok=False, path=output_path, size=0, status=0, error=last_error)


def manifest_entry_for_local_files(day: date) -> dict:
    iso_date = day.isoformat()
    dhw_name = build_filename("dhw", day)
    hs_name = build_filename("hs", day)
    dhw_path = DHW_DIR / dhw_name
    hs_path = HS_DIR / hs_name

    dhw_size = file_size_if_non_empty(dhw_path)
    hs_size = file_size_if_non_empty(hs_path)

    ok = dhw_size > 0 and hs_size > 0
    error = ""
    if not ok:
        missing = []
        if dhw_size == 0:
            missing.append("dhw missing")
        if hs_size == 0:
            missing.append("hs missing")
        error = "; ".join(missing)

    return {
        "dhw": relative_repo_path(dhw_path),
        "hs": relative_repo_path(hs_path),
        "ok": ok,
        "error": error,
        "sizes": {"dhw": int(dhw_size), "hs": int(hs_size)},
        "date": iso_date,
    }


def process_date(
    day: date,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
    user_agent: str,
) -> tuple[str, dict]:
    iso_date = day.isoformat()
    dhw_name = build_filename("dhw", day)
    hs_name = build_filename("hs", day)
    dhw_path = DHW_DIR / dhw_name
    hs_path = HS_DIR / hs_name

    dhw_result = download_file_with_retries(
        url=build_url("dhw", day),
        output_path=dhw_path,
        timeout_seconds=timeout_seconds,
        retries=retries,
        backoff_seconds=backoff_seconds,
        user_agent=user_agent,
    )
    hs_result = download_file_with_retries(
        url=build_url("hs", day),
        output_path=hs_path,
        timeout_seconds=timeout_seconds,
        retries=retries,
        backoff_seconds=backoff_seconds,
        user_agent=user_agent,
    )

    dhw_size = file_size_if_non_empty(dhw_path)
    hs_size = file_size_if_non_empty(hs_path)
    ok = dhw_size > 0 and hs_size > 0

    errors: list[str] = []
    if not ok:
        if dhw_result.error:
            errors.append(f"dhw: {dhw_result.error}")
        if hs_result.error:
            errors.append(f"hs: {hs_result.error}")
        if dhw_size == 0 and not dhw_result.error:
            errors.append("dhw: missing local file")
        if hs_size == 0 and not hs_result.error:
            errors.append("hs: missing local file")

    entry = {
        "dhw": relative_repo_path(dhw_path),
        "hs": relative_repo_path(hs_path),
        "ok": ok,
        "error": "; ".join(errors) if errors else "",
        "sizes": {"dhw": int(dhw_size), "hs": int(hs_size)},
    }
    return iso_date, entry


def write_manifest_atomic(manifest: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=True, sort_keys=False)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def build_manifest(start_date: date, end_date: date, by_date: dict[str, dict]) -> dict:
    ordered_dates = sorted(by_date.keys())
    ordered_by_date = {iso_date: by_date[iso_date] for iso_date in ordered_dates}
    ok_dates = [iso_date for iso_date in ordered_dates if ordered_by_date[iso_date].get("ok")]

    return {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "ok_dates": ok_dates,
        "by_date": ordered_by_date,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily NOAA DHW + HotSpot files and generate noaa_manifest.json"
    )
    parser.add_argument("--start", required=True, type=parse_iso_date, help="start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, type=parse_iso_date, help="end date (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=6, help="parallel worker count (default: 6)")
    parser.add_argument("--timeout", type=float, default=60.0, help="request timeout seconds (default: 60)")
    parser.add_argument("--retries", type=int, default=4, help="retry count after first attempt (default: 4)")
    parser.add_argument("--backoff", type=float, default=1.5, help="base backoff seconds (default: 1.5)")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="skip network downloads and regenerate manifest from local files only",
    )
    parser.add_argument(
        "--user-agent",
        default="coral-bleaching-tracker-noaa-downloader/1.0",
        help="http user agent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start > args.end:
        raise SystemExit("--start must be on or before --end")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DHW_DIR.mkdir(parents=True, exist_ok=True)
    HS_DIR.mkdir(parents=True, exist_ok=True)

    days = date_range_inclusive(args.start, args.end)
    by_date: dict[str, dict] = {}

    if args.manifest_only:
        for day in days:
            entry = manifest_entry_for_local_files(day)
            iso_date = entry.pop("date")
            by_date[iso_date] = entry
    else:
        workers = max(1, int(args.workers))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    process_date,
                    day,
                    float(args.timeout),
                    int(args.retries),
                    float(args.backoff),
                    str(args.user_agent),
                ): day
                for day in days
            }

            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                iso_date, entry = future.result()
                by_date[iso_date] = entry
                completed += 1
                if completed % 50 == 0 or completed == total:
                    print(f"progress: {completed}/{total}")

    manifest = build_manifest(args.start, args.end, by_date)
    write_manifest_atomic(manifest, MANIFEST_PATH)

    ok_count = len(manifest["ok_dates"])
    total_count = len(manifest["by_date"])
    print(f"wrote manifest: {MANIFEST_PATH.as_posix()}")
    print(f"ok dates: {ok_count}/{total_count}")


if __name__ == "__main__":
    main()
