# main.py

This file is part of the modularized brain tumor classification project.
import os
from src.data_loader import get_data_generators
from src.model_builder import build_model
from src.train_model import compute_class_weights, train_model
from src.evaluator import evaluate_and_generate_outputs

train_dir = os.path.expanduser("~/Downloads/Training")
test_dir = os.path.expanduser("~/Downloads/Testing")

train_data, test_data = get_data_generators(train_dir, test_dir)
class_weights = compute_class_weights(train_data)

model, base_model = build_model()

# Train frozen layers
train_model(model, train_data, test_data, class_weights, fine_tune=False)

# Fine-tune the base model
base_model.trainable = True
train_model(model, train_data, test_data, fine_tune=True)

# Save model
model.save("outputs/brain_tumor_model.keras")

# Evaluate and visualize
evaluate_and_generate_outputs(model, test_data)
