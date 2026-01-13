# run_model.py
import json
import os
from cpu_cultural_enrich import generate_base_caption
from cultural_context import CulturalContextController

IMAGE_DIR = "eval"
OUT_FILE = "model_output.json"

esn = CulturalContextController()
results = {}

for img in sorted(os.listdir(IMAGE_DIR)):
    img_id = img.split(".")[0]
    path = os.path.join(IMAGE_DIR, img)

    caption = generate_base_caption(path).strip()
    mode, conf = esn.predict_mode(caption)

    results[img_id] = {
        "base_caption": caption,
        "final_caption": caption,
        "mode": mode,
        "esn_confidence": round(conf, 3)
    }

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("✅ model_output.json generated")
