import os, sys, argparse, json, pickle
from pathlib import Path
from PIL import Image
import numpy as np

# set TF env before importing TF to reduce logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# robust load_model import
try:
    from tensorflow.keras.models import load_model
except Exception:
    try:
        from keras.models import load_model
    except Exception as e:
        raise ImportError("Install tensorflow or keras in this environment") from e

PROJ = Path(__file__).resolve().parent
MODEL_PATHS = ["model.keras", "model.h5", "traffic_signal_model.h5", "traffic_signal_model.keras"]

def find_model():
    for p in MODEL_PATHS:
        fp = PROJ / p
        if fp.exists():
            return str(fp)
    return None

def preprocess(img: Image.Image):
    # must match training: resize->RGB->grayscale by average->normalize (pixel-128)/128
    img = img.convert("RGB").resize((32,32), Image.BILINEAR)
    arr = np.array(img).astype("float32")
    arr = np.sum(arr/3.0, axis=2, keepdims=True)    # average grayscale
    arr = (arr - 128.0) / 128.0
    return arr.reshape(1,32,32,1)

def load_classes():
    cj = PROJ / "classes.json"
    if not cj.exists():
        return None
    with open(cj, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(i): v for i,v in enumerate(data)}
    if isinstance(data, dict):
        return {str(k): v for k,v in data.items()}
    return None

def predict_image(model, classes_map, img_path):
    img = Image.open(img_path)
    x = preprocess(img)
    probs = model.predict(x)[0]
    idxs = probs.argsort()[::-1][:5].astype(int)
    for i in idxs:
        name = classes_map.get(str(i), f"Class_{i}")
        print(f"{i}\t{name}\t{probs[i]*100:.2f}%")
    # save preprocessed visualization for inspection
    px = ((x.squeeze()*128.0)+128.0).clip(0,255).astype(np.uint8).squeeze()
    Image.fromarray(px).save(PROJ/"debug_preprocessed.png")
    print("Saved preprocessed image -> debug_preprocessed.png (visual check)")

def quick_test_eval(model):
    # if DataSource/test.p exists, compute accuracy + confusion matrix
    tp = PROJ / "DataSource" / "test.p"
    if not tp.exists():
        print("No DataSource/test.p available for batch evaluation.")
        return
    with open(tp, "rb") as f:
        test = pickle.load(f)
    X_test, y_test = test['features'], test['labels']
    # apply same preprocessing for whole test set
    Xg = np.sum(X_test/3.0, axis=3, keepdims=True)
    Xg = (Xg - 128.0) / 128.0
    preds = model.predict(Xg, verbose=0)
    pred_classes = preds.argmax(axis=1)
    acc = (pred_classes == y_test).mean()
    print(f"Test set accuracy: {acc*100:.2f}% ({len(y_test)} samples)")
    # show some misclassified examples
    mis = np.where(pred_classes != y_test)[0][:12]
    if len(mis):
        print("First misclassified (index, true, pred):")
        for i in mis:
            print(i, y_test[i], pred_classes[i])
    else:
        print("No misclassifications in first 12 samples.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", help="Single image path to predict", default=None)
    parser.add_argument("--eval-test", action="store_true", help="Run quick evaluation on DataSource/test.p")
    args = parser.parse_args()

    model_file = find_model()
    if model_file is None:
        print("No model file found in project root. Put model.keras or model.h5 next to this script.")
        sys.exit(1)

    print("Loading model:", model_file)
    model = load_model(model_file)
    print("Model summary:")
    model.summary()

    classes_map = load_classes()
    if classes_map is None:
        print("classes.json missing or invalid. Create a list of 43 labels in classes.json.")
    else:
        print("Loaded classes.json with", len(classes_map), "entries")

    if args.image:
        predict_image(model, classes_map or {}, args.image)

    if args.eval_test:
        quick_test_eval(model)

if __name__ == "__main__":
    main()