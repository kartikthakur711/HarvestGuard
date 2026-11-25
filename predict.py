import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parent

def load_classifier():
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
    model = keras.models.load_model(ROOT / "models" / "harvestguard.keras", compile=False)
    return model, metadata

def predict(image, model, metadata, top_k=3):
    height, width = metadata["image_size"]
    pixels = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    scores = np.asarray(model.predict(pixels[None, ...], verbose=0))[0]
    labels = metadata["display_class_names"]
    if scores.shape != (len(labels),):
        raise ValueError("Model output does not match metadata")
    return [(labels[i], float(scores[i])) for i in np.argsort(scores)[::-1][:top_k]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    model, metadata = load_classifier()
    with Image.open(args.image) as image:
        for label, score in predict(image, model, metadata):
            print(f"{label}: {score:.2%}")
