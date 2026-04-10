#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo


BASE_DIR = Path("/srv/analema")
NORMALIZED_DIR = BASE_DIR / "normalized" / "sessions"
PROCESSED_DIR = BASE_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"

FRAME_INTERVAL_SECONDS = 10
THUMB_WIDTH = 320


@dataclass
class Session:
    session_id: str
    station_id: str
    camera_id: str
    target: str
    start_at_local: datetime
    end_at_local: datetime
    date_local: str
    timezone: str
    raw_abs_path: Path
    raw_rel_path: str
    json_path: Path


def load_session(path: Path) -> Session:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return Session(
        session_id=data["session_id"],
        station_id=data["station_id"],
        camera_id=data["camera_id"],
        target=data["target"],
        start_at_local=datetime.fromisoformat(data["start_at_local"]),
        end_at_local=datetime.fromisoformat(data["end_at_local"]),
        date_local=data["date_local"],
        timezone=data["timezone"],
        raw_abs_path=BASE_DIR / data["raw_rel_path"],
        raw_rel_path=data["raw_rel_path"],
        json_path=path,
    )


def iter_sessions() -> Iterator[Path]:
    yield from sorted(NORMALIZED_DIR.glob("*/*/*.json"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def session_output_dir(session: Session) -> Path:
    start_clock = session.start_at_local.strftime("%H-%M-%S")
    return (
        PROCESSED_DIR
        / session.station_id
        / session.camera_id
        / session.date_local
        / start_clock
    )


def build_frame_id(session: Session, captured_at_local: datetime) -> str:
    return (
        f"{session.station_id}__{session.camera_id}__"
        f"{captured_at_local.date().isoformat()}__"
        f"{captured_at_local.strftime('%H-%M-%S')}"
    )


def iter_capture_times(start: datetime, end: datetime, step_seconds: int) -> Iterator[datetime]:
    current = start
    while current < end:
        yield current
        current += timedelta(seconds=step_seconds)


def extract_single_frame(input_path: Path, offset_seconds: int, output_path: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(offset_seconds),
        "-i", str(input_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0 and output_path.exists()


def create_thumbnail(input_image: Path, output_thumb: Path, width: int = THUMB_WIDTH) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_image),
        "-vf", f"scale={width}:-1",
        "-frames:v", "1",
        "-update", "1",
        str(output_thumb),
    ]
    subprocess.run(cmd, check=True)

def write_summary(summary_path: Path, payload: dict) -> None:
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_ndjson(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def already_extracted(session: Session) -> bool:
    out_dir = session_output_dir(session)
    frames_ndjson = out_dir / "metadata" / "frames.ndjson"
    return frames_ndjson.exists()


def process_session(session: Session) -> None:
    if not session.raw_abs_path.exists():
        print(f"[WARN] Arquivo bruto nao encontrado: {session.raw_abs_path}")
        return

    out_dir = session_output_dir(session)
    frames_dir = out_dir / "frames"
    thumbs_dir = out_dir / "thumbs"
    metadata_dir = out_dir / "metadata"
    derived_dir = out_dir / "derived"

    ensure_dir(frames_dir)
    ensure_dir(thumbs_dir)
    ensure_dir(metadata_dir)
    ensure_dir(derived_dir)

    records: list[dict] = []

    for captured_at_local in iter_capture_times(
        session.start_at_local,
        session.end_at_local,
        FRAME_INTERVAL_SECONDS,
    ):
        offset_seconds = int((captured_at_local - session.start_at_local).total_seconds())
        clock_str = captured_at_local.strftime("%H-%M-%S")

        frame_filename = f"{clock_str}.jpg"
        thumb_filename = f"{clock_str}.jpg"

        frame_path = frames_dir / frame_filename
        thumb_path = thumbs_dir / thumb_filename

        frame_ok = True
        if not frame_path.exists():
            frame_ok = extract_single_frame(session.raw_abs_path, offset_seconds, frame_path)

        if not frame_ok:
            continue

            
        if not thumb_path.exists():
            create_thumbnail(frame_path, thumb_path)

        frame_id = build_frame_id(session, captured_at_local)

        records.append({
            "frame_id": frame_id,
            "session_id": session.session_id,
            "station_id": session.station_id,
            "camera_id": session.camera_id,
            "target": session.target,
            "captured_at_local": captured_at_local.isoformat(),
            "clock_time": captured_at_local.strftime("%H:%M:%S"),
            "date_local": session.date_local,
            "image_rel_path": str(frame_path.relative_to(BASE_DIR)),
            "thumbnail_rel_path": str(thumb_path.relative_to(BASE_DIR)),
            "sun_detected": None,
            "sun_px_x": None,
            "sun_px_y": None,
            "sun_blob_area": None,
            "disk_visibility_score": None,
            "saturation_score": None,
            "sharpness_score": None,
            "cloud_obstruction_score": None,
            "artifact_score": None,
            "quality_score": None,
            "quality_label": None,
            "quality_model_version": "solar_quality_v1",
            "usable_for_series": None,
            "processing_status": "extracted"
        })

    frames_ndjson = metadata_dir / "frames.ndjson"
    summary_json = metadata_dir / "summary.json"

    append_ndjson(frames_ndjson, records)

    summary_payload = {
        "session_id": session.session_id,
        "station_id": session.station_id,
        "camera_id": session.camera_id,
        "target": session.target,
        "date_local": session.date_local,
        "start_at_local": session.start_at_local.isoformat(),
        "end_at_local": session.end_at_local.isoformat(),
        "frame_interval_seconds": FRAME_INTERVAL_SECONDS,
        "frame_count": len(records),
        "status": "frames_extracted"
    }
    write_summary(summary_json, summary_payload)

    print(f"[OK] Frames extraidos: {session.session_id} ({len(records)} frames)")


def main() -> None:
    ensure_dir(LOGS_DIR)

    total = 0
    processed = 0

    for session_json_path in iter_sessions():
        total += 1
        session = load_session(session_json_path)

        if already_extracted(session):
            print(f"[SKIP] Sessao ja extraida: {session.session_id}")
            continue

        process_session(session)
        processed += 1

    print(f"[INFO] Sessoes encontradas: {total}")
    print(f"[INFO] Sessoes processadas agora: {processed}")


if __name__ == "__main__":
    main()
