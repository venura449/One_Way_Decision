# Simple YOLO vehicle-count server

Two endpoints (upload image → get counts):
- `POST /count_total` returns **total vehicles** only.
- `POST /count_by_type` returns **car/motorcycle/bus/truck + total**.
- `POST /priority_decision` accepts **two images** and returns which lane has priority + gate open time.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

## Example request (PowerShell)

```powershell
$img = "C:\path\to\frame.jpg"

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/count_by_type" `
  -Form @{ image = Get-Item $img }
```

## Example request (2 images, priority)

```powershell
$a = "C:\path\to\laneA.jpg"
$b = "C:\path\to\laneB.jpg"

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/priority_decision" `
  -Form @{ image_a = Get-Item $a; image_b = Get-Item $b }
```

## Notes

- Vehicle classes are: car, motorcycle, bus, truck (COCO ids: 2, 3, 5, 7).
- The first run will download a YOLO model (`yolov8n.pt`) automatically.
- Set detection threshold with `CONF` (default `0.25`) and model with `YOLO_MODEL`.
- Gate time policy is configurable with `CAR_SEC`, `MOTORCYCLE_SEC`, `BUS_SEC`, `TRUCK_SEC`, plus `MIN_GATE_SEC` and `MAX_GATE_SEC`.
- If you have custom models for winner/time, place them in `Model/models/`:
  - `winner_classifier.pkl`
  - `gate_seconds_regressor.pkl`
