import os
from typing import Dict, Tuple

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO)
NAME_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes (decode failed).")
    return img


def _count_vehicles(model: YOLO, bgr: np.ndarray, conf: float) -> Tuple[Dict[str, int], float]:
    t0 = cv2.getTickCount()
    results = model.predict(bgr, conf=conf, verbose=False)
    r0 = results[0]

    counts: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "total": 0}
    if r0.boxes is not None and r0.boxes.cls is not None:
        clses = r0.boxes.cls.detach().cpu().numpy().astype(int)
        for cls_id in clses:
            cls_id = int(cls_id)
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            label = NAME_MAP[cls_id]
            counts[label] += 1
            counts["total"] += 1

    ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
    return counts, ms


app = FastAPI()

MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")
CONF = float(os.environ.get("CONF", "0.25"))
model = YOLO(MODEL)

CUSTOM_MODEL_DIR = os.environ.get("CUSTOM_MODEL_DIR", os.path.join("Model", "models"))
WINNER_MODEL_PATH = os.environ.get(
    "WINNER_MODEL_PATH", os.path.join(CUSTOM_MODEL_DIR, "winner_classifier.pkl")
)
GATE_MODEL_PATH = os.environ.get(
    "GATE_MODEL_PATH", os.path.join(CUSTOM_MODEL_DIR, "gate_seconds_regressor.pkl")
)

try:
    winner_model = joblib.load(WINNER_MODEL_PATH)
except Exception:
    winner_model = None

try:
    gate_model = joblib.load(GATE_MODEL_PATH)
except Exception:
    gate_model = None

CAR_SEC = float(os.environ.get("CAR_SEC", "2.0"))
MOTORCYCLE_SEC = float(os.environ.get("MOTORCYCLE_SEC", "1.5"))
BUS_SEC = float(os.environ.get("BUS_SEC", "3.0"))
TRUCK_SEC = float(os.environ.get("TRUCK_SEC", "3.5"))
MIN_GATE_SEC = float(os.environ.get("MIN_GATE_SEC", "3.0"))
MAX_GATE_SEC = float(os.environ.get("MAX_GATE_SEC", "60.0"))


def _gate_time_seconds(counts: Dict[str, int]) -> float:
    raw = (
        counts.get("car", 0) * CAR_SEC
        + counts.get("motorcycle", 0) * MOTORCYCLE_SEC
        + counts.get("bus", 0) * BUS_SEC
        + counts.get("truck", 0) * TRUCK_SEC
    )
    return float(max(MIN_GATE_SEC, min(MAX_GATE_SEC, raw)))


def _features_from_counts(counts: Dict[str, int]) -> np.ndarray:
    # Keep this order stable; must match how your .pkl models were trained.
    return np.array(
        [
            counts.get("car", 0),
            counts.get("motorcycle", 0),
            counts.get("bus", 0),
            counts.get("truck", 0),
            counts.get("total", 0),
        ],
        dtype=np.float32,
    ).reshape(1, -1)


def _expected_n_features(m):
    n = getattr(m, "n_features_in_", None)
    try:
        return int(n) if n is not None else None
    except Exception:
        return None


def _winner_features_17(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> np.ndarray:
    """
    17 dims = 10 lane-count features + 7 constants.

    Layout:
    - 5 counts for A: car,motorcycle,bus,truck,total
    - 5 counts for B: car,motorcycle,bus,truck,total
    - 7 constants WIN_CONST_1..WIN_CONST_7 (defaults 0)
    """
    xa = _features_from_counts(counts_a).astype(np.float32)  # (1,5)
    xb = _features_from_counts(counts_b).astype(np.float32)  # (1,5)
    consts = np.array(
        [
            [
                float(os.environ.get("WIN_CONST_1", "0")),
                float(os.environ.get("WIN_CONST_2", "0")),
                float(os.environ.get("WIN_CONST_3", "0")),
                float(os.environ.get("WIN_CONST_4", "0")),
                float(os.environ.get("WIN_CONST_5", "0")),
                float(os.environ.get("WIN_CONST_6", "0")),
                float(os.environ.get("WIN_CONST_7", "0")),
            ]
        ],
        dtype=np.float32,
    )  # (1,7)
    return np.concatenate([xa, xb, consts], axis=1)  # (1,17)


def _fit_feature_shape(x: np.ndarray, expected):
    if expected is None:
        return x
    if x.shape[1] == expected:
        return x
    if x.shape[1] > expected:
        return x[:, :expected]
    pad = np.zeros((x.shape[0], expected - x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=1)


def _predict_winner(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> str:
    """
    Custom model path:
    - If `winner_model` exists: use it to choose winner.
    Fallback:
    - higher total vehicles wins, else tie.
    """
    if winner_model is not None:
        expected = _expected_n_features(winner_model)
        x = _winner_features_17(counts_a, counts_b)
        x = _fit_feature_shape(x, expected)
        try:
            pred = winner_model.predict(x)[0]
            pred = str(pred).lower()
            if pred in {"a", "lane_a", "left"}:
                return "A"
            if pred in {"b", "lane_b", "right"}:
                return "B"
            if pred in {"tie", "equal"}:
                return "tie"
        except Exception:
            pass

    if counts_a["total"] > counts_b["total"]:
        return "A"
    if counts_b["total"] > counts_a["total"]:
        return "B"
    return "tie"


def _predict_gate_seconds(counts: Dict[str, int]) -> float:
    """
    Custom model path:
    - If `gate_model` exists: use it to predict seconds.
    Fallback:
    - rule-based `_gate_time_seconds`.
    """
    if gate_model is not None:
        x = _features_from_counts(counts)
        try:
            y = float(gate_model.predict(x)[0])
            if y == y:  # not NaN
                return float(max(MIN_GATE_SEC, min(MAX_GATE_SEC, y)))
        except Exception:
            pass
    return _gate_time_seconds(counts)


@app.post("/count_total")
async def count_total(image: UploadFile = File(...)):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        bgr = _decode_image_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    counts, ms = _count_vehicles(model, bgr, conf=CONF)
    return {"total": counts["total"], "inference_ms": round(ms, 2)}


@app.post("/count_by_type")
async def count_by_type(image: UploadFile = File(...)):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        bgr = _decode_image_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    counts, ms = _count_vehicles(model, bgr, conf=CONF)
    return {**counts, "inference_ms": round(ms, 2)}


@app.post("/priority_decision")
async def priority_decision(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    a_bytes = await image_a.read()
    b_bytes = await image_b.read()
    if not a_bytes or not b_bytes:
        raise HTTPException(status_code=400, detail="Both images are required")

    try:
        a_bgr = _decode_image_bytes(a_bytes)
        b_bgr = _decode_image_bytes(b_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    counts_a, ms_a = _count_vehicles(model, a_bgr, conf=CONF)
    counts_b, ms_b = _count_vehicles(model, b_bgr, conf=CONF)

    winner = _predict_winner(counts_a, counts_b)
    winner_counts = counts_a if winner in {"A", "tie"} else counts_b
    gate_sec = _predict_gate_seconds(winner_counts)

    return {
        "winner": winner,
        "gate_open_seconds": round(gate_sec, 2),
        "lane_a": {**counts_a, "inference_ms": round(ms_a, 2)},
        "lane_b": {**counts_b, "inference_ms": round(ms_b, 2)},
        "policy": {
            "custom_models": {
                "winner_model_path": WINNER_MODEL_PATH,
                "gate_model_path": GATE_MODEL_PATH,
                "winner_model_loaded": winner_model is not None,
                "gate_model_loaded": gate_model is not None,
            },
            "seconds_per_vehicle": {
                "car": CAR_SEC,
                "motorcycle": MOTORCYCLE_SEC,
                "bus": BUS_SEC,
                "truck": TRUCK_SEC,
            },
            "min_gate_seconds": MIN_GATE_SEC,
            "max_gate_seconds": MAX_GATE_SEC,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

