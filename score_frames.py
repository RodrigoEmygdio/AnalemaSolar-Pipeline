#!/usr/bin/env python3

import json
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path("/srv/analema")
PROCESSED_DIR = BASE_DIR / "processed"

QUALITY_MODEL_VERSION = "solar_quality_v3"


def variance_of_laplacian(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def apply_exclusion_mask(gray: np.ndarray) -> np.ndarray:
    """
    Mascara regioes nao cientificas do frame:
    - margens extremas
    - timestamp no topo direito
    - watermark no canto inferior esquerdo
    """
    masked = gray.copy()
    h, w = masked.shape

    # 1. margens globais
    margin_x = int(w * 0.03)
    margin_y = int(h * 0.03)

    masked[:margin_y, :] = 0
    masked[h - margin_y:, :] = 0
    masked[:, :margin_x] = 0
    masked[:, w - margin_x:] = 0

    # 2. timestamp no topo direito
    ts_x0 = int(w * 0.68)
    ts_y1 = int(h * 0.14)
    masked[:ts_y1, ts_x0:] = 0

    # 3. watermark no canto inferior esquerdo
    wm_x1 = int(w * 0.28)
    wm_y0 = int(h * 0.86)
    masked[wm_y0:, :wm_x1] = 0

    return masked


def detect_bright_source(gray: np.ndarray):
    """
    Detecta a maior regiao brilhante do frame mascarado.
    """
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return False, None, thresh

    largest = max(contours, key=cv2.contourArea)
    return True, largest, thresh


def contour_circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return 0.0

    circ = 4 * np.pi * area / (perimeter * perimeter)
    return float(max(0.0, min(circ, 1.0)))


def compute_area_ratio(contour, gray: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    total = gray.shape[0] * gray.shape[1]
    return float(area / total)


def compute_bbox_aspect_ratio(contour) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return 999.0
    return float(w / h)


def compute_centroid(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None, None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def centroid_is_in_valid_region(cx: int, cy: int, shape) -> bool:
    h, w = shape

    if cx is None or cy is None:
        return False

    if cy < int(h * 0.12):
        return False
    if cy > int(h * 0.97):
        return False
    if cx < int(w * 0.03):
        return False
    if cx > int(w * 0.97):
        return False

    return True


def saturation_score(gray: np.ndarray) -> float:
    saturated_pixels = np.sum(gray >= 250)
    total_pixels = gray.size
    return float(saturated_pixels / total_pixels)


def global_cloud_score(gray: np.ndarray) -> float:
    std = np.std(gray)
    score = 1.0 - (std / 128.0)
    return float(min(max(score, 0.0), 1.0))


def compute_edge_strength(gray: np.ndarray, contour) -> float:
    x, y, w, h = cv2.boundingRect(contour)

    pad = 10
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, gray.shape[1])
    y1 = min(y + h + pad, gray.shape[0])

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    edges = cv2.Canny(roi, 80, 160)
    edge_ratio = np.count_nonzero(edges) / edges.size

    score = min(edge_ratio / 0.08, 1.0)
    return float(max(0.0, score))


def compute_artifact_score(gray: np.ndarray, contour) -> float:
    area_ratio = compute_area_ratio(contour, gray)
    sat = saturation_score(gray)

    if area_ratio < 0.001 and sat > 0.01:
        return 0.8

    if area_ratio < 0.002 and sat > 0.005:
        return 0.5

    return 0.1


def label_quality(score: float) -> str:
    if score >= 0.80:
        return "excellent"
    if score >= 0.60:
        return "good"
    if score >= 0.40:
        return "usable"
    return "discard"


def is_valid_solar_disk(
    area_ratio: float,
    circularity: float,
    bbox_aspect_ratio: float,
    edge_strength: float,
    cx: int,
    cy: int,
    shape
) -> bool:
    # area minima mais rigida
    if area_ratio < 0.0010:
        return False

    # area maxima absurda
    if area_ratio > 0.08:
        return False

    # pouco circular
    if circularity < 0.55:
        return False

    # bbox muito alongada
    if not (0.70 <= bbox_aspect_ratio <= 1.30):
        return False

    # borda fraca demais
    if edge_strength < 0.08:
        return False

    # centro em regiao invalida
    if not centroid_is_in_valid_region(cx, cy, shape):
        return False

    return True


def score_valid_solar_frame(
    disk_visibility_score: float,
    sharpness_score: float,
    cloud_obstruction_score: float,
    saturation_score_value: float,
    artifact_score: float
) -> float:
    quality = (
        0.30 * disk_visibility_score +
        0.20 * sharpness_score +
        0.20 * (1.0 - cloud_obstruction_score) +
        0.15 * (1.0 - saturation_score_value) +
        0.15 * (1.0 - artifact_score)
    )
    return float(min(max(quality, 0.0), 1.0))


def process_ndjson(path: Path):
    updated = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            img_path = BASE_DIR / record["image_rel_path"]
            image = cv2.imread(str(img_path))

            if image is None:
                record["bright_source_detected"] = False
                record["sun_detected"] = False
                record["quality_score"] = 0.0
                record["quality_label"] = "discard"
                record["quality_model_version"] = QUALITY_MODEL_VERSION
                updated.append(record)
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            analysis_gray = apply_exclusion_mask(gray)

            bright_detected, contour, _ = detect_bright_source(analysis_gray)
            record["bright_source_detected"] = bool(bright_detected)

            if not bright_detected:
                record["sun_detected"] = False
                record["sun_px_x"] = None
                record["sun_px_y"] = None
                record["sun_blob_area"] = None
                record["disk_visibility_score"] = 0.0
                record["saturation_score"] = float(saturation_score(gray))
                record["sharpness_score"] = 0.0
                record["cloud_obstruction_score"] = float(global_cloud_score(gray))
                record["artifact_score"] = 0.0
                record["edge_strength_score"] = 0.0
                record["area_ratio"] = 0.0
                record["bbox_aspect_ratio"] = None
                record["quality_score"] = 0.0
                record["quality_label"] = "discard"
                record["quality_model_version"] = QUALITY_MODEL_VERSION
                updated.append(record)
                continue

            area = float(cv2.contourArea(contour))
            circularity = contour_circularity(contour)
            area_ratio = compute_area_ratio(contour, analysis_gray)
            bbox_aspect_ratio = compute_bbox_aspect_ratio(contour)
            edge_strength = compute_edge_strength(analysis_gray, contour)
            sat = float(saturation_score(gray))
            cloud = float(global_cloud_score(gray))
            artifact = float(compute_artifact_score(analysis_gray, contour))
            sharp = float(min(variance_of_laplacian(gray) / 500.0, 1.0))

            cx, cy = compute_centroid(contour)

            record["sun_px_x"] = cx
            record["sun_px_y"] = cy
            record["sun_blob_area"] = area
            record["edge_strength_score"] = edge_strength
            record["area_ratio"] = area_ratio
            record["bbox_aspect_ratio"] = bbox_aspect_ratio
            record["saturation_score"] = sat
            record["sharpness_score"] = sharp
            record["cloud_obstruction_score"] = cloud
            record["artifact_score"] = artifact
            record["quality_model_version"] = QUALITY_MODEL_VERSION

            valid_disk = is_valid_solar_disk(
                area_ratio=area_ratio,
                circularity=circularity,
                bbox_aspect_ratio=bbox_aspect_ratio,
                edge_strength=edge_strength,
                cx=cx,
                cy=cy,
                shape=analysis_gray.shape
            )

            if not valid_disk:
                record["sun_detected"] = False
                record["disk_visibility_score"] = 0.0
                record["quality_score"] = 0.0
                record["quality_label"] = "discard"
                updated.append(record)
                continue

            record["sun_detected"] = True
            record["disk_visibility_score"] = float(circularity)

            quality = score_valid_solar_frame(
                disk_visibility_score=record["disk_visibility_score"],
                sharpness_score=sharp,
                cloud_obstruction_score=cloud,
                saturation_score_value=sat,
                artifact_score=artifact,
            )

            record["quality_score"] = quality
            record["quality_label"] = label_quality(quality)

            updated.append(record)

    with open(path, "w", encoding="utf-8") as f:
        for r in updated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    for path in PROCESSED_DIR.glob("*/*/*/*/metadata/frames.ndjson"):
        print("scoring:", path)
        process_ndjson(path)


if __name__ == "__main__":
    main()