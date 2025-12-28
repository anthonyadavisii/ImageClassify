import onnxruntime as ort
import numpy as np
from PIL import Image
import json

# Paths
MODEL_PATH = r"C:\<path to model.onnx"
LABELS_PATH = r"C:\<path to imagenet_labels.json>"	 # 1k labels, one per line
IMG_PATH = input("Enter image file path: ")

def load_labels(path):
    with open(path, "r") as f:
        return json.load(f)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0  # scale to 0–1
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # NCHW
    return img
def classify_image(image_path):
    # Prefer QNN, fall back to CPU
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["QNNExecutionProvider", "CPUExecutionProvider"]
    )
    print("Active providers:", session.get_providers())
    labels = load_labels(LABELS_PATH)
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.astype(np.float32)
    input_name = session.get_inputs()[0].name
    print("Input Shape:", session.get_inputs()[0].shape)
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})[0]
    print("Output shape:", outputs.shape)
    print("Raw output:", outputs)
    # Softmax
    probs = np.exp(outputs) / np.sum(np.exp(outputs))
    # Top‑5
    top5_idx = probs[0].argsort()[-5:][::-1]
    print("\nTop‑5 predictions:")
    for idx in top5_idx:
        print(f"{labels[idx]} — {probs[0][idx]:.4f}")

# Example usage
if __name__ == "__main__":
    classify_image(IMG_PATH)
