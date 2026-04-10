#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/srv/analema")
PROCESSED_DIR = BASE_DIR / "processed"
SITE_DIR = BASE_DIR / "publish/site"

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def iter_frames():
    yield from PROCESSED_DIR.glob("*/*/*/*/metadata/frames.ndjson")


def load_all_frames():

    frames = []

    for path in iter_frames():

        with open(path) as f:

            for line in f:

                frames.append(json.loads(line))

    return frames


def group_by_date(frames):

    result = defaultdict(list)

    for f in frames:

        result[f["date_local"]].append(f)

    return result


def group_by_time(frames):

    result = defaultdict(list)

    for f in frames:

        result[f["clock_time"]].append(f)

    return result


def write_index(frames):

    index_path = SITE_DIR / "index.html"

    html = f"""
    <html>
    <body>

    <h1>Analema Solar</h1>

    <p>Total frames: {len(frames)}</p>

    <a href="/dates/">Datas</a><br>
    <a href="/times/">Horarios</a>

    </body>
    </html>
    """

    index_path.write_text(html)


def write_dates(dates):

    root = SITE_DIR / "dates"

    ensure_dir(root)

    for date, frames in dates.items():

        page = root / date

        ensure_dir(page)

        html = "<html><body>"

        html += f"<h1>{date}</h1>"

        for f in frames:

            html += f"""
            <img src="/{f['thumbnail_rel_path']}" />
            """

        html += "</body></html>"

        (page / "index.html").write_text(html)


def write_times(times):

    root = SITE_DIR / "times"

    ensure_dir(root)

    for t, frames in times.items():

        page = root / t.replace(":", "-")

        ensure_dir(page)

        html = "<html><body>"

        html += f"<h1>{t}</h1>"

        for f in frames:

            html += f"""
            <img src="/{f['thumbnail_rel_path']}" />
            """

        html += "</body></html>"

        (page / "index.html").write_text(html)


def main():

    ensure_dir(SITE_DIR)

    frames = load_all_frames()

    write_index(frames)

    write_dates(group_by_date(frames))

    write_times(group_by_time(frames))


if __name__ == "__main__":
    main()