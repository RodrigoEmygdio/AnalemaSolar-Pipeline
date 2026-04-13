#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path("/srv/analema")
PROCESSED_DIR = BASE_DIR / "processed"
PUBLISH_DIR = BASE_DIR / "publish"
SITE_DIR = PUBLISH_DIR / "site"
DATA_DIR = SITE_DIR / "data"

SITE_TITLE = "Analema Solar - Grupo Modulação - Estação Brasilia"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_site_dir() -> None:
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    ensure_dir(SITE_DIR)
    ensure_dir(DATA_DIR)


def iter_frames_ndjson():
    yield from sorted(PROCESSED_DIR.glob("*/*/*/*/metadata/frames.ndjson"))


def iter_summary_json():
    yield from sorted(PROCESSED_DIR.glob("*/*/*/*/metadata/summary.json"))


def load_ndjson(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_frames() -> list[dict]:
    frames: list[dict] = []

    for path in iter_frames_ndjson():
        frames.extend(load_ndjson(path))

    return frames


def collect_summaries() -> list[dict]:
    summaries: list[dict] = []

    for path in iter_summary_json():
        summaries.append(load_json(path))

    return summaries


def group_by_date(frames: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        grouped[frame["date_local"]].append(frame)

    for date in grouped:
        grouped[date] = sorted(grouped[date], key=lambda x: (x["clock_time"], x["session_id"]))

    return dict(sorted(grouped.items()))


def group_by_time(frames: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        grouped[frame["clock_time"]].append(frame)

    for clock_time in grouped:
        grouped[clock_time] = sorted(grouped[clock_time], key=lambda x: (x["date_local"], x["session_id"]))

    return dict(sorted(grouped.items()))


def group_summaries_by_date(summaries: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for summary in summaries:
        grouped[summary["date_local"]].append(summary)

    for date in grouped:
        grouped[date] = sorted(grouped[date], key=lambda x: x["start_at_local"])

    return dict(sorted(grouped.items()))


def copy_asset_if_needed(rel_path: str) -> str:
    """
    Copia frame/thumb do /srv/analema para dentro do site estático e retorna o caminho relativo dentro do site.
    """
    source = BASE_DIR / rel_path
    target = SITE_DIR / rel_path

    ensure_dir(target.parent)
    if source.exists() and not target.exists():
        shutil.copy2(source, target)

    return rel_path


def html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #fafafa;
      color: #222;
    }}
    h1, h2, h3 {{
      margin-top: 1.2em;
    }}
    a {{
      color: #0b57d0;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .topnav {{
      margin-bottom: 20px;
    }}
    .topnav a {{
      margin-right: 14px;
    }}
    .meta {{
      color: #555;
      font-size: 0.95em;
    }}
    .session {{
      background: #fff;
      border: 1px solid #ddd;
      padding: 12px;
      margin: 12px 0;
      border-radius: 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 10px;
    }}
    .card img {{
      width: 100%;
      height: auto;
      border-radius: 6px;
      display: block;
      margin-bottom: 8px;
      background: #eee;
    }}
    .small {{
      font-size: 0.9em;
      color: #555;
      line-height: 1.4;
    }}
    .index-list li {{
      margin: 6px 0;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      background: #eee;
      font-size: 0.85em;
      margin-left: 6px;
    }}
  </style>
</head>
<body>
  <div class="topnav">
    <a href="/index.html">Início</a>
    <a href="/dates/index.html">Datas</a>
    <a href="/times/index.html">Horários</a>
  </div>
  {body}
</body>
</html>
"""


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def build_home(frames: list[dict], summaries: list[dict]) -> None:
    dates = sorted({f["date_local"] for f in frames})
    times = sorted({f["clock_time"] for f in frames})

    latest_date = dates[-1] if dates else "-"
    body = f"""
    <h1>{SITE_TITLE}</h1>
    <p class="meta">Site observacional estático gerado automaticamente.</p>

    <h2>Resumo</h2>
    <ul class="index-list">
      <li>Total de sessões: <strong>{len(summaries)}</strong></li>
      <li>Total de frames: <strong>{len(frames)}</strong></li>
      <li>Total de datas: <strong>{len(dates)}</strong></li>
      <li>Total de horários distintos: <strong>{len(times)}</strong></li>
      <li>Última data disponível: <strong>{latest_date}</strong></li>
    </ul>

    <h2>Navegação</h2>
    <ul class="index-list">
      <li><a href="/dates/index.html">Explorar por data</a></li>
      <li><a href="/times/index.html">Explorar por horário (HH:MM:SS)</a></li>
    </ul>
    """

    write_text(SITE_DIR / "index.html", html_page(SITE_TITLE, body))


def build_dates_index(grouped_dates: dict[str, list[dict]]) -> None:
    items = []
    for date, frames in grouped_dates.items():
        items.append(f'<li><a href="/dates/{date}/index.html">{date}</a> <span class="pill">{len(frames)} frames</span></li>')

    body = f"""
    <h1>Datas disponíveis</h1>
    <ul class="index-list">
      {''.join(items)}
    </ul>
    """

    write_text(SITE_DIR / "dates" / "index.html", html_page("Datas", body))


def build_times_index(grouped_times: dict[str, list[dict]]) -> None:
    items = []
    for clock_time, frames in grouped_times.items():
        slug = clock_time.replace(":", "-")
        items.append(f'<li><a href="/times/{slug}/index.html">{clock_time}</a> <span class="pill">{len(frames)} frames</span></li>')

    body = f"""
    <h1>Horários disponíveis</h1>
    <ul class="index-list">
      {''.join(items)}
    </ul>
    """

    write_text(SITE_DIR / "times" / "index.html", html_page("Horários", body))


def render_frame_card(frame: dict) -> str:
    thumb_rel = copy_asset_if_needed(frame["thumbnail_rel_path"])
    img_rel = copy_asset_if_needed(frame["image_rel_path"])

    quality = frame.get("quality_score")
    quality_str = "-" if quality is None else f"{quality:.3f}"

    return f"""
    <div class="card">
      <a href="/{img_rel}" target="_blank">
        <img src="/{thumb_rel}" alt="{frame['frame_id']}">
      </a>
      <div class="small">
        <div><strong>Data:</strong> {frame['date_local']}</div>
        <div><strong>Hora:</strong> {frame['clock_time']}</div>
        <div><strong>Sessão:</strong> {frame['session_id']}</div>
        <div><strong>Alvo:</strong> {frame['target']}</div>
        <div><strong>Sol detectado:</strong> {frame.get('sun_detected')}</div>
        <div><strong>Qualidade:</strong> {quality_str}</div>
        <div><strong>Modelo:</strong> {frame.get('quality_model_version')}</div>
      </div>
    </div>
    """


def build_date_pages(grouped_dates: dict[str, list[dict]], summaries_by_date: dict[str, list[dict]]) -> None:
    for date, frames in grouped_dates.items():
        cards = "".join(render_frame_card(frame) for frame in frames)

        summaries_html = ""
        for summary in summaries_by_date.get(date, []):
            summaries_html += f"""
            <div class="session">
              <div><strong>Sessão:</strong> {summary['session_id']}</div>
              <div><strong>Início:</strong> {summary['start_at_local']}</div>
              <div><strong>Fim inferido:</strong> {summary['end_at_local']}</div>
              <div><strong>Fim efetivo:</strong> {summary.get('effective_end_at_local', '-')}</div>
              <div><strong>Frames:</strong> {summary.get('frame_count', '-')}</div>
              <div><strong>Falhas:</strong> {summary.get('failed_frames', '-')}</div>
            </div>
            """

        body = f"""
        <h1>Data {date}</h1>
        <p class="meta">Frames agrupados por data observacional.</p>

        <h2>Sessões do dia</h2>
        {summaries_html or '<p>Nenhuma sessão encontrada.</p>'}

        <h2>Frames</h2>
        <div class="grid">
          {cards}
        </div>
        """

        write_text(SITE_DIR / "dates" / date / "index.html", html_page(f"Data {date}", body))


def build_time_pages(grouped_times: dict[str, list[dict]]) -> None:
    for clock_time, frames in grouped_times.items():
        slug = clock_time.replace(":", "-")
        cards = "".join(render_frame_card(frame) for frame in frames)

        body = f"""
        <h1>Horário {clock_time}</h1>
        <p class="meta">Comparação do mesmo horário ao longo das datas disponíveis.</p>

        <div class="grid">
          {cards}
        </div>
        """

        write_text(SITE_DIR / "times" / slug / "index.html", html_page(f"Horário {clock_time}", body))


def build_json_indexes(frames: list[dict], grouped_dates: dict[str, list[dict]], grouped_times: dict[str, list[dict]]) -> None:
    ensure_dir(DATA_DIR / "by_date")
    ensure_dir(DATA_DIR / "by_time")

    latest_payload = {
        "total_frames": len(frames),
        "total_dates": len(grouped_dates),
        "total_times": len(grouped_times),
        "latest_date": max(grouped_dates.keys()) if grouped_dates else None
    }
    write_text(DATA_DIR / "latest.json", json.dumps(latest_payload, ensure_ascii=False, indent=2))

    for date, date_frames in grouped_dates.items():
        write_text(DATA_DIR / "by_date" / f"{date}.json", json.dumps(date_frames, ensure_ascii=False, indent=2))

    for clock_time, time_frames in grouped_times.items():
        slug = clock_time.replace(":", "-")
        payload = {
            "clock_time": clock_time,
            "frames": time_frames,
        }
        write_text(DATA_DIR / "by_time" / f"{slug}.json", json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    reset_site_dir()

    frames = collect_frames()
    summaries = collect_summaries()

    grouped_dates = group_by_date(frames)
    grouped_times = group_by_time(frames)
    summaries_by_date = group_summaries_by_date(summaries)

    build_home(frames, summaries)
    build_dates_index(grouped_dates)
    build_times_index(grouped_times)
    build_date_pages(grouped_dates, summaries_by_date)
    build_time_pages(grouped_times)
    build_json_indexes(frames, grouped_dates, grouped_times)

    print(f"[OK] Site estático gerado em: {SITE_DIR}")


if __name__ == "__main__":
    main()