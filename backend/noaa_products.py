from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from backend.config import NOAA_DHW_DIR, NOAA_HS_DIR


@dataclass(frozen=True)
class NoaaProductSpec:
    kind: str
    directory: Path
    base_url: str
    filename_template: str
    variable_name: str
    file_pattern: re.Pattern[str]


PRODUCT_SPECS: dict[str, NoaaProductSpec] = {
    "dhw": NoaaProductSpec(
        kind="dhw",
        directory=NOAA_DHW_DIR,
        base_url=(
            "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/"
            "5km/v3.1_op/nc/v1.0/daily/dhw"
        ),
        filename_template="ct5km_dhw_v3.1_{ymd}.nc",
        variable_name="degree_heating_week",
        file_pattern=re.compile(r"^ct5km_dhw_v3\.1_(\d{8})\.nc$"),
    ),
    "hs": NoaaProductSpec(
        kind="hs",
        directory=NOAA_HS_DIR,
        base_url=(
            "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/"
            "5km/v3.1_op/nc/v1.0/daily/hs"
        ),
        filename_template="ct5km_hs_v3.1_{ymd}.nc",
        variable_name="hotspot",
        file_pattern=re.compile(r"^ct5km_hs_v3\.1_(\d{8})\.nc$"),
    ),
}


def product_kinds() -> tuple[str, ...]:
    return tuple(PRODUCT_SPECS.keys())


def get_product_spec(kind: str) -> NoaaProductSpec:
    try:
        return PRODUCT_SPECS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown NOAA product kind: {kind}") from exc


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def normalize_iso_date(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return parse_iso_date(text).isoformat()
    except ValueError:
        return None


def build_filename(kind: str, day: date) -> str:
    return get_product_spec(kind).filename_template.format(ymd=day.strftime("%Y%m%d"))


def build_product_path(kind: str, day: date) -> Path:
    spec = get_product_spec(kind)
    return spec.directory / build_filename(kind, day)


def build_product_url(kind: str, day: date) -> str:
    spec = get_product_spec(kind)
    return f"{spec.base_url}/{day.strftime('%Y')}/{build_filename(kind, day)}"


def parse_product_date_from_filename(kind: str, filename: str) -> str | None:
    match = get_product_spec(kind).file_pattern.match(filename)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()


def enumerate_mondays(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        return []
    offset = (7 - start_date.weekday()) % 7
    first_monday = start_date + timedelta(days=offset)
    if first_monday > end_date:
        return []
    total_weeks = ((end_date - first_monday).days // 7) + 1
    return [first_monday + timedelta(days=7 * index) for index in range(total_weeks)]


def date_range_to_iso(dates: Iterable[date]) -> list[str]:
    return [day.isoformat() for day in dates]
