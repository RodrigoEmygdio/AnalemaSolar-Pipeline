from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATASET_DIR = Path("dataset")
OUTPUT_CSV = Path("solar_calibration_v2_frame_results.csv")
OUTPUT_JSON = Path("solar_calibration_v2_results.json")
SAVE_DEBUG_IMAGES = True
DEBUG_DIRNAME = "_debug"

# ----------------------------
# ROI de busca do Sol
# Frações da imagem: x0, y0, x1, y1
# Ajuste conforme a geometria real da tua estação
# Exemplo atual: evita cantos e overlays
# ----------------------------
USE_SEARCH_ROI = True
SEARCH_ROI = (0.10, 0.08, 0.90, 0.88)

# ----------------------------
# Máscaras dos overlays da câmera
# Frações da imagem
# timestamp: canto superior direito
# watermark: canto inferior esquerdo
# ----------------------------
MASK_TIMESTAMP = True
TIMESTAMP_RECT = (0.73, 0.00, 1.00, 0.12)

MASK_WATERMARK = True
WATERMARK_RECT = (0.00, 0.82, 0.28, 1.00)

# Thresholds
BRIGHT_THRESHOLD = 240
SATURATION_THRESHOLD = 250
HALO_THRESHOLD = 160
GHOST_THRESHOLD = 180

# Áreas mínimas
MIN_DISK_AREA = 40
MIN_GHOST_AREA = 8

# Dilatação para anel do halo
HALO_RING_RADIUS = 20

# Pesos do score
W_CIRCULARITY = 4.0
W_SATURATION = -3.0
W_HALO = -2.0
W_GHOST = -2.0
W_DIAMETER = -1.0
W_EDGE_SHARPNESS = 3.0

# Penalização de centro instável entre frames da mesma config
STABILITY_DIVISOR = 20.0


# ============================================================
# MODELOS
# ============================================================

@dataclass
class FrameMetrics:
    config_name: str
    round_name: str
    file_name: str
    disk_found: bool

    disk_area: float = 0.0
    disk_perimeter: float = 0.0
    circularity: float = 0.0
    equivalent_diameter: float = 0.0

    saturated_area: float = 0.0
    saturation_ratio_inside_disk: float = 0.0

    halo_area: float = 0.0
    halo_ratio: float = 0.0

    ghost_count: int = 0
    ghost_total_area: float = 0.0

    edge_sharpness: float = 0.0

    centroid_x: float = 0.0
    centroid_y: float = 0.0

    score: float = -999999.0
    rejection_reason: Optional[str] = None


# ============================================================
# AUXILIARES MATEMÁTICOS
# ============================================================

def calc_circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def equivalent_diameter(area: float) -> float:
    if area <= 0:
        return 0.0
    return float(math.sqrt(4.0 * area / math.pi))


def extract_round_name(config_name: str) -> str:
    """
    Extrai a rodada de uma configuração.
    A1 -> A
    B3 -> B
    gamma_test_1 -> gamma_test_1 (sem prefixo simples)
    """
    m = re.match(r"^([A-Za-z]+)", config_name)
    if m:
        return m.group(1).upper()
    return config_name


# ============================================================
# IO / IMAGEM
# ============================================================

def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Falha ao ler imagem: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ============================================================
# MÁSCARAS GEOMÉTRICAS
# ============================================================

def frac_rect_to_pixels(
    shape: tuple[int, int],
    rect_frac: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    h, w = shape
    x0f, y0f, x1f, y1f = rect_frac
    x0 = max(0, min(w, int(round(x0f * w))))
    y0 = max(0, min(h, int(round(y0f * h))))
    x1 = max(0, min(w, int(round(x1f * w))))
    y1 = max(0, min(h, int(round(y1f * h))))
    return x0, y0, x1, y1


def create_exclusion_mask(shape: tuple[int, int]) -> np.ndarray:
    """
    255 = região válida
    0   = região excluída
    """
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)

    if MASK_TIMESTAMP:
        x0, y0, x1, y1 = frac_rect_to_pixels(shape, TIMESTAMP_RECT)
        mask[y0:y1, x0:x1] = 0

    if MASK_WATERMARK:
        x0, y0, x1, y1 = frac_rect_to_pixels(shape, WATERMARK_RECT)
        mask[y0:y1, x0:x1] = 0

    return mask


def create_search_roi_mask(shape: tuple[int, int]) -> np.ndarray:
    """
    255 = região onde é permitido procurar o Sol
    0   = fora da ROI
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if not USE_SEARCH_ROI:
        mask[:, :] = 255
        return mask

    x0, y0, x1, y1 = frac_rect_to_pixels(shape, SEARCH_ROI)
    mask[y0:y1, x0:x1] = 255
    return mask


def combine_valid_mask(shape: tuple[int, int]) -> np.ndarray:
    exclusion = create_exclusion_mask(shape)
    roi = create_search_roi_mask(shape)
    return cv2.bitwise_and(exclusion, roi)


def create_disk_mask(shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, k, iterations=1)


# ============================================================
# CONTORNOS
# ============================================================

def largest_valid_contour(mask: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None
    return max(valid, key=cv2.contourArea)


def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return 0.0, 0.0
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


# ============================================================
# DETECÇÃO DO DISCO
# ============================================================

def detect_solar_disk(gray: np.ndarray, valid_mask: np.ndarray) -> Optional[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # aplica máscara válida
    masked = cv2.bitwise_and(blur, blur, mask=valid_mask)

    _, bright = cv2.threshold(masked, BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)

    # força exclusão fora da máscara válida
    bright = cv2.bitwise_and(bright, valid_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=1)

    contour = largest_valid_contour(bright, MIN_DISK_AREA)
    return contour


# ============================================================
# MÉTRICA DE NITIDEZ DE BORDA
# ============================================================

def compute_edge_sharpness(gray: np.ndarray, disk_mask: np.ndarray) -> float:
    """
    Mede o gradiente médio ao redor da borda do disco.
    Quanto maior, mais abrupta a borda.
    """
    inner = cv2.erode(
        disk_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    outer = cv2.dilate(
        disk_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    border_band = cv2.subtract(outer, inner)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    values = grad[border_band > 0]
    if values.size == 0:
        return 0.0

    return float(np.mean(values) / 255.0)


# ============================================================
# GHOSTS
# ============================================================

def count_ghosts(
    gray: np.ndarray,
    valid_mask: np.ndarray,
    disk_mask: np.ndarray,
    main_contour: np.ndarray,
) -> tuple[int, float]:
    ghost_mask = np.zeros_like(gray, dtype=np.uint8)
    ghost_mask[gray >= GHOST_THRESHOLD] = 255
    ghost_mask = cv2.bitwise_and(ghost_mask, valid_mask)

    contours, _ = cv2.findContours(ghost_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ghost_count = 0
    ghost_total_area = 0.0
    main_area = cv2.contourArea(main_contour)

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_GHOST_AREA:
            continue

        tmp = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(tmp, [c], -1, 255, thickness=-1)

        overlap = cv2.bitwise_and(tmp, disk_mask)
        overlap_area = cv2.countNonZero(overlap)

        if overlap_area > 0.5 * area:
            continue

        if area < 0.05 * main_area or overlap_area == 0:
            ghost_count += 1
            ghost_total_area += float(area)

    return ghost_count, ghost_total_area


# ============================================================
# ANÁLISE DO FRAME
# ============================================================

def analyze_frame(image_path: Path, config_name: str, save_debug: bool = False) -> FrameMetrics:
    img = load_image(image_path)
    gray = to_gray(img)
    h, w = gray.shape

    round_name = extract_round_name(config_name)
    valid_mask = combine_valid_mask((h, w))

    metrics = FrameMetrics(
        config_name=config_name,
        round_name=round_name,
        file_name=image_path.name,
        disk_found=False,
    )

    contour = detect_solar_disk(gray, valid_mask)
    if contour is None:
        metrics.rejection_reason = "disk_not_found"
        return metrics

    metrics.disk_found = True

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circ = calc_circularity(area, perimeter)
    eq_diam = equivalent_diameter(area)
    cx, cy = contour_centroid(contour)

    metrics.disk_area = float(area)
    metrics.disk_perimeter = float(perimeter)
    metrics.circularity = float(circ)
    metrics.equivalent_diameter = float(eq_diam)
    metrics.centroid_x = float(cx)
    metrics.centroid_y = float(cy)

    disk_mask = create_disk_mask((h, w), contour)

    # Saturação no disco
    sat_mask = np.zeros_like(gray, dtype=np.uint8)
    sat_mask[gray >= SATURATION_THRESHOLD] = 255
    sat_mask = cv2.bitwise_and(sat_mask, valid_mask)

    sat_inside = cv2.bitwise_and(sat_mask, disk_mask)
    saturated_area = cv2.countNonZero(sat_inside)
    disk_pixels = cv2.countNonZero(disk_mask)

    metrics.saturated_area = float(saturated_area)
    metrics.saturation_ratio_inside_disk = float(saturated_area / disk_pixels) if disk_pixels > 0 else 0.0

    # Halo
    expanded = dilate_mask(disk_mask, HALO_RING_RADIUS)
    ring = cv2.subtract(expanded, disk_mask)

    halo_mask = np.zeros_like(gray, dtype=np.uint8)
    halo_mask[gray >= HALO_THRESHOLD] = 255
    halo_mask = cv2.bitwise_and(halo_mask, valid_mask)

    halo_in_ring = cv2.bitwise_and(halo_mask, ring)
    halo_area = cv2.countNonZero(halo_in_ring)
    ring_area = cv2.countNonZero(ring)

    metrics.halo_area = float(halo_area)
    metrics.halo_ratio = float(halo_area / ring_area) if ring_area > 0 else 0.0

    # Ghosts
    ghost_count, ghost_total_area = count_ghosts(gray, valid_mask, disk_mask, contour)
    metrics.ghost_count = int(ghost_count)
    metrics.ghost_total_area = float(ghost_total_area)

    # Nitidez de borda
    metrics.edge_sharpness = compute_edge_sharpness(gray, disk_mask)

    # Score final do frame
    metrics.score = float(
        (W_CIRCULARITY * metrics.circularity)
        + (W_SATURATION * metrics.saturation_ratio_inside_disk)
        + (W_HALO * metrics.halo_ratio)
        + (W_GHOST * metrics.ghost_count)
        + (W_DIAMETER * (metrics.equivalent_diameter / 100.0))
        + (W_EDGE_SHARPNESS * metrics.edge_sharpness)
    )

    if save_debug:
        debug_dir = image_path.parent / DEBUG_DIRNAME
        debug_dir.mkdir(parents=True, exist_ok=True)

        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # região válida de busca
        valid_border = np.zeros_like(gray, dtype=np.uint8)
        valid_border[valid_mask > 0] = 255
        contours_valid, _ = cv2.findContours(valid_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, contours_valid, -1, (255, 255, 0), 1)

        # contorno do disco
        cv2.drawContours(dbg, [contour], -1, (0, 255, 0), 2)
        cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        # overlays mascarados
        exclusion_mask = create_exclusion_mask((h, w))
        excluded = cv2.bitwise_not(exclusion_mask)
        overlay_excluded = dbg.copy()
        overlay_excluded[excluded > 0] = (80, 80, 80)
        dbg = cv2.addWeighted(overlay_excluded, 0.35, dbg, 0.65, 0)

        # halo
        overlay_halo = dbg.copy()
        overlay_halo[halo_in_ring > 0] = (255, 255, 0)
        dbg = cv2.addWeighted(overlay_halo, 0.30, dbg, 0.70, 0)

        # saturação
        overlay_sat = dbg.copy()
        overlay_sat[sat_inside > 0] = (0, 0, 255)
        dbg = cv2.addWeighted(overlay_sat, 0.40, dbg, 0.60, 0)

        # ghosts
        ghost_mask = np.zeros_like(gray, dtype=np.uint8)
        ghost_mask[gray >= GHOST_THRESHOLD] = 255
        ghost_mask = cv2.bitwise_and(ghost_mask, valid_mask)
        ghost_only = ghost_mask.copy()
        ghost_only[disk_mask > 0] = 0

        overlay_ghost = dbg.copy()
        overlay_ghost[ghost_only > 0] = (255, 0, 255)
        dbg = cv2.addWeighted(overlay_ghost, 0.25, dbg, 0.75, 0)

        text_lines = [
            f"cfg={config_name}",
            f"round={round_name}",
            f"score={metrics.score:.4f}",
            f"circ={metrics.circularity:.4f}",
            f"diam={metrics.equivalent_diameter:.2f}",
            f"sat={metrics.saturation_ratio_inside_disk:.4f}",
            f"halo={metrics.halo_ratio:.4f}",
            f"ghosts={metrics.ghost_count}",
            f"edge={metrics.edge_sharpness:.4f}",
        ]

        y = 24
        for line in text_lines:
            cv2.putText(
                dbg,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24

        out_path = debug_dir / f"debug_{image_path.name}"
        cv2.imwrite(str(out_path), dbg)

    return metrics


# ============================================================
# AGREGAÇÃO
# ============================================================

def summarize_by_config(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["disk_found"] == True].copy()
    if valid.empty:
        return pd.DataFrame()

    grouped = valid.groupby("config_name", as_index=False).agg(
        round_name=("round_name", "first"),
        frame_count=("file_name", "count"),
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        mean_circularity=("circularity", "mean"),
        mean_diameter=("equivalent_diameter", "mean"),
        mean_saturation=("saturation_ratio_inside_disk", "mean"),
        mean_halo=("halo_ratio", "mean"),
        mean_ghost_count=("ghost_count", "mean"),
        mean_ghost_area=("ghost_total_area", "mean"),
        mean_edge_sharpness=("edge_sharpness", "mean"),
        centroid_x_std=("centroid_x", "std"),
        centroid_y_std=("centroid_y", "std"),
    )

    grouped["std_score"] = grouped["std_score"].fillna(0.0)
    grouped["centroid_x_std"] = grouped["centroid_x_std"].fillna(0.0)
    grouped["centroid_y_std"] = grouped["centroid_y_std"].fillna(0.0)

    grouped["stability_penalty"] = (
        grouped["centroid_x_std"] + grouped["centroid_y_std"]
    ) / STABILITY_DIVISOR

    grouped["final_score"] = grouped["mean_score"] - grouped["stability_penalty"]

    grouped = grouped.sort_values(
        by=["final_score", "mean_circularity", "mean_edge_sharpness"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return grouped


def summarize_by_round(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["disk_found"] == True].copy()
    if valid.empty:
        return pd.DataFrame()

    grouped = valid.groupby("round_name", as_index=False).agg(
        config_count=("config_name", "nunique"),
        frame_count=("file_name", "count"),
        mean_score=("score", "mean"),
        mean_circularity=("circularity", "mean"),
        mean_diameter=("equivalent_diameter", "mean"),
        mean_saturation=("saturation_ratio_inside_disk", "mean"),
        mean_halo=("halo_ratio", "mean"),
        mean_ghost_count=("ghost_count", "mean"),
        mean_edge_sharpness=("edge_sharpness", "mean"),
    )

    grouped = grouped.sort_values(by="mean_score", ascending=False).reset_index(drop=True)
    return grouped


# ============================================================
# COLETA
# ============================================================

def collect_image_paths(dataset_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []

    for config_dir in sorted(dataset_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        if config_dir.name.startswith("_"):
            continue

        for image_path in sorted(config_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            items.append((config_dir.name, image_path))

    return items


# ============================================================
# PIPELINE
# ============================================================

def main() -> None:
    if not DATASET_DIR.exists():
        raise SystemExit(f"Pasta não encontrada: {DATASET_DIR.resolve()}")

    image_items = collect_image_paths(DATASET_DIR)
    if not image_items:
        raise SystemExit("Nenhuma imagem encontrada na estrutura esperada.")

    all_metrics: list[FrameMetrics] = []

    for config_name, image_path in image_items:
        try:
            metrics = analyze_frame(
                image_path=image_path,
                config_name=config_name,
                save_debug=SAVE_DEBUG_IMAGES,
            )
            all_metrics.append(metrics)
            print(
                f"[OK] {config_name}/{image_path.name} "
                f"disk_found={metrics.disk_found} score={metrics.score:.4f}"
            )
        except Exception as exc:
            print(f"[ERRO] {config_name}/{image_path.name}: {exc}")
            all_metrics.append(
                FrameMetrics(
                    config_name=config_name,
                    round_name=extract_round_name(config_name),
                    file_name=image_path.name,
                    disk_found=False,
                    rejection_reason=f"exception: {exc}",
                )
            )

    frame_df = pd.DataFrame([asdict(m) for m in all_metrics])
    frame_df.to_csv(OUTPUT_CSV, index=False)

    config_summary = summarize_by_config(frame_df)
    round_summary = summarize_by_round(frame_df)

    result_payload = {
        "frame_results": [asdict(m) for m in all_metrics],
        "config_ranking": config_summary.to_dict(orient="records"),
        "round_summary": round_summary.to_dict(orient="records"),
    }

    OUTPUT_JSON.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== RANKING POR CONFIGURAÇÃO ===")
    if config_summary.empty:
        print("Nenhuma configuração válida conseguiu detecção do disco.")
    else:
        print(config_summary.to_string(index=False))

    print("\n=== RESUMO POR RODADA ===")
    if round_summary.empty:
        print("Nenhuma rodada com frames válidos.")
    else:
        print(round_summary.to_string(index=False))

    print(f"\nCSV salvo em: {OUTPUT_CSV.resolve()}")
    print(f"JSON salvo em: {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()