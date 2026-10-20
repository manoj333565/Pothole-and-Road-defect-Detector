
from pathlib import Path
import yaml
from ultralytics import YOLO

print("Starting Pothole Detection Training...")

ROOT = Path(__file__).resolve().parent
data_yaml = ROOT / "Pothole.v1-raw.yolov8" / "data.yaml"

if not data_yaml.is_file():
    raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

with open(data_yaml, "r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

print("Dataset YAML:", data_yaml)
print("Dataset configuration:", data_config)

if not isinstance(data_config, dict) or "train" not in data_config or "val" not in data_config:
    raise ValueError("Dataset YAML must define train and val paths.")

model = YOLO("yolov8n.pt")

results = model.train(
    data=str(data_yaml.resolve()),
    epochs=30,
    imgsz=416,
    batch=2,
    patience=8,
    workers=0,
    cache=False,
    device="cpu",
    project="runs/pothole",
    name="pothole_img416_fixed",
    plots=True,
    save=True,
    exist_ok=False
)

print("Training completed!")
