# src/evaluator.py

This file is part of the modularized brain tumor classification project.

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from .gradcam import make_gradcam_heatmap, save_gradcam

def evaluate_and_generate_outputs(model, test_data, output_dir="outputs/gradcam_outputs"):
    filenames = test_data.filenames
    true_labels = test_data.classes
    pred_probs = model.predict(test_data)
    pred_labels = np.argmax(pred_probs, axis=1)
    confidences = np.max(pred_probs, axis=1)
    class_names = list(test_data.class_indices.keys())

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, fname in enumerate(filenames):
        true_label = class_names[true_labels[i]]
        pred_label = class_names[pred_labels[i]]
        conf = confidences[i]

        img_path = os.path.join(test_data.directory, fname)
        try:
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            heatmap = make_gradcam_heatmap(img_array, model, "Conv_1")
            save_path = os.path.join(output_dir, f"{i}_{pred_label}.jpg")
            save_gradcam(img_path, heatmap, save_path)

            results.append({
                "filename": fname,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": conf
            })
        except Exception as e:
            print(f" Failed to process {fname}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/prediction_results.csv", index=False)
