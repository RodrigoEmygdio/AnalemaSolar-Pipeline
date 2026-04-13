#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo


BASE_DIR = Path("/srv/analema")
CONFIG_DIR = BASE_DIR / "config"
INCOMING_DIR = BASE_DIR / "incoming" / "upload"
RAW_DIR = BASE_DIR / "raw"
NORMALIZED_DIR = BASE_DIR / "normalized" / "sessions"
LOGS_DIR = BASE_DIR / "logs"

NORMALIZER_VERSION = "1.1"

INTELBRAS_NVR_REGEX = re.compile(
    r"^NVR_(ch\d+)_(main|extra|sub)_(\d{14})_(\d{14})\.dav$"
)


@dataclass
class Station:
    station_id: str
    station_name: str
    timezone: str
    source_hosts: list[str]
    raw: dict


@dataclass
class Camera:
    camera_id: str
    station_id: str
    target: str
    source_host: str
    channel: str
    stream_type: str
    parser_name: str
    raw: dict


def load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_stations() -> Dict[str, Station]:
    result: Dict[str, Station] = {}
    stations_dir = CONFIG_DIR / "stations"

    for path in stations_dir.glob("*.json"):
        data = load_json_file(path)
        station = Station(
            station_id=data["station_id"],
            station_name=data["station_name"],
            timezone=data["timezone"],
            source_hosts=data.get("source_hosts", []),
            raw=data,
        )
        result[station.station_id] = station

    return result


def load_cameras() -> Dict[str, Camera]:
    result: Dict[str, Camera] = {}
    cameras_dir = CONFIG_DIR / "cameras"

    for path in cameras_dir.glob("*.json"):
        data = load_json_file(path)
        camera = Camera(
            camera_id=data["camera_id"],
            station_id=data["station_id"],
            target=data["target"],
            source_host=data["source_host"],
            channel=data["channel"],
            stream_type=data["stream_type"],
            parser_name=data["parser_name"],
            raw=data,
        )
        result[camera.camera_id] = camera

    return result


def find_station_by_source_host(source_host: str, stations: Dict[str, Station]) -> Optional[Station]:
    for station in stations.values():
        if source_host in station.source_hosts:
            return station
    return None


def find_camera(source_host: str, channel: str, stream_type: str, cameras: Dict[str, Camera]) -> Optional[Camera]:
    for camera in cameras.values():
        if (
            camera.source_host == source_host
            and camera.channel == channel
            and camera.stream_type == stream_type
        ):
            return camera
    return None


def parse_intelbras_filename(filename: str) -> Optional[dict]:
    match = INTELBRAS_NVR_REGEX.match(filename)
    if not match:
        return None

    channel, stream_type, start_raw, end_raw = match.groups()
    return {
        "channel": channel,
        "stream_type": stream_type,
        "start_raw": start_raw,
        "end_raw": end_raw,
    }


def parse_timestamp(raw_ts: str, tz_name: str) -> datetime:
    naive = datetime.strptime(raw_ts, "%Y%m%d%H%M%S")
    return naive.replace(tzinfo=ZoneInfo(tz_name))


def iso_compact(dt: datetime) -> str:
    return dt.isoformat().replace(":", "-")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_session_id(station_id: str, camera_id: str, start_at_local: datetime) -> str:
    return f"{station_id}__{camera_id}__{iso_compact(start_at_local)}"


def build_raw_target_path(station_id: str, camera_id: str, date_local: str, filename: str) -> Path:
    return RAW_DIR / station_id / camera_id / date_local / filename


def build_session_json_path(date_local: str, session_id: str) -> Path:
    year = date_local[:4]
    return NORMALIZED_DIR / year / date_local / f"{session_id}.json"


def normalize_one_file(file_path: Path, stations: Dict[str, Station], cameras: Dict[str, Camera]) -> Optional[dict]:
    """
    file_path esperado:
    /srv/analema/incoming/upload/<source_host>/<date_dir>/<filename>
    """
    try:
        source_host = file_path.parents[1].name
        source_date_dir = file_path.parents[0].name
        filename = file_path.name
    except IndexError:
        print(f"[WARN] Caminho inesperado: {file_path}")
        return None

    parsed = parse_intelbras_filename(filename)
    if not parsed:
        print(f"[WARN] Nome de arquivo nao reconhecido: {filename}")
        return None

    station = find_station_by_source_host(source_host, stations)
    if not station:
        print(f"[WARN] Nenhuma estacao encontrada para source_host={source_host}")
        return None

    camera = find_camera(source_host, parsed["channel"], parsed["stream_type"], cameras)
    if not camera:
        print(
            f"[WARN] Nenhuma camera encontrada para "
            f"source_host={source_host}, channel={parsed['channel']}, stream_type={parsed['stream_type']}"
        )
        return None

    start_at_local = parse_timestamp(parsed["start_raw"], station.timezone)
    end_at_local = parse_timestamp(parsed["end_raw"], station.timezone)
    date_local = start_at_local.date().isoformat()

    session_id = build_session_id(station.station_id, camera.camera_id, start_at_local)
    session_json_path = build_session_json_path(date_local, session_id)

    if session_json_path.exists():
        print(f"[SKIP] Sessao ja normalizada: {session_id}")
        return None

    raw_target_path = build_raw_target_path(
        station.station_id,
        camera.camera_id,
        date_local,
        filename,
    )
    ensure_dir(raw_target_path.parent)

    if not raw_target_path.exists():
        shutil.copy2(file_path, raw_target_path)

    raw_size = raw_target_path.stat().st_size
    duration_seconds_inferred = int((end_at_local - start_at_local).total_seconds())
    normalized_at = datetime.now(ZoneInfo(station.timezone)).isoformat()

    session_payload = {
        "session_id": session_id,
        "station_id": station.station_id,
        "camera_id": camera.camera_id,
        "target": camera.target,
        "source_type": "nvr_upload",
        "source_host": source_host,
        "source_date_dir": source_date_dir,
        "raw_filename": filename,
        "channel": parsed["channel"],
        "stream_type": parsed["stream_type"],
        "start_at_local": start_at_local.isoformat(),
        "end_at_local": end_at_local.isoformat(),
        "date_local": date_local,
        "timezone": station.timezone,
        "raw_rel_path": str(raw_target_path.relative_to(BASE_DIR)),
        "raw_file_size_bytes": raw_size,
        "duration_seconds_inferred": duration_seconds_inferred,
        "normalized_at": normalized_at,
        "processing_status": "normalized",
        "parser_name": camera.parser_name,
        "parser_version": "1.0",
        "normalizer_version": NORMALIZER_VERSION
    }

    ensure_dir(session_json_path.parent)

    with session_json_path.open("w", encoding="utf-8") as f:
        json.dump(session_payload, f, ensure_ascii=False, indent=2)

    return session_payload


def iter_incoming_media() -> list[Path]:
    return sorted(INCOMING_DIR.glob("*/*/*.dav"))


def main() -> None:
    ensure_dir(LOGS_DIR)
    ensure_dir(RAW_DIR)
    ensure_dir(NORMALIZED_DIR)

    stations = load_stations()
    cameras = load_cameras()

    files = iter_incoming_media()
    if not files:
        print("[INFO] Nenhum arquivo .dav encontrado em incoming/upload.")
        return

    normalized_count = 0

    for file_path in files:
        try:
            result = normalize_one_file(file_path, stations, cameras)
            if result:
                normalized_count += 1
                print(f"[OK] Sessao normalizada: {result['session_id']}")
        except Exception as exc:
            print(f"[ERROR] Falha ao normalizar {file_path}: {exc}")

    print(f"[INFO] Total de sessoes normalizadas nesta execucao: {normalized_count}")


if __name__ == "__main__":
    main()