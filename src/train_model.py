# src/train_model.py

This file is part of the modularized brain tumor classification project.

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

def compute_class_weights(train_data):
    labels = train_data.classes
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))

def train_model(model, train_data, test_data, class_weights=None, fine_tune=False):
    learning_rate = 1e-5 if fine_tune else 1e-4
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_data, validation_data=test_data, epochs=10,
                        class_weight=class_weights, callbacks=[early_stop])
    return history

