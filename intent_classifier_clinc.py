import json
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

JSON_PATH = "./data_full.json"

MODEL_PATH = "./intent_model_clinc.keras"
LABEL_ENCODER_PATH = "./intent_label_classes.npy"
SELECTED_INTENTS_PATH = "./selected_intents.txt"

MAX_TOKENS = 10000
MAX_LEN = 32
BATCH_SIZE = 32
EPOCHS = 15

USE_SUBSET = False # change to true if we want to analyze a subset

SELECTED_INTENTS: List[str] = [] # you can add the subsets you choose here


def combine_split(data: dict, split_name: str, include_oos: bool = True):
    base = data.get(split_name, [])
    if include_oos:
        oos_key = f"oos_{split_name}"
        base = base + data.get(oos_key, [])

    texts = np.array([row[0] for row in base], dtype="object")
    intents = np.array([row[1] for row in base], dtype="object")
    return texts, intents


def load_clinc_json(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_texts, train_intents = combine_split(data, "train", include_oos=True)
    val_texts, val_intents = combine_split(data, "val", include_oos=True)
    test_texts, test_intents = combine_split(data, "test", include_oos=True)

    print("Loaded CLINC data_full.json:")
    print("  train:", train_texts.shape[0])
    print("  val:  ", val_texts.shape[0])
    print("  test: ", test_texts.shape[0])

    print("\nExample train sample:")
    print("  text:", train_texts[0])
    print("  intent:", train_intents[0])

    return (
        train_texts,
        train_intents,
        val_texts,
        val_intents,
        test_texts,
        test_intents,
    )


def load_selected_intents_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [x for x in lines if x]


def maybe_filter_subset(
    train_texts: np.ndarray,
    train_intents: np.ndarray,
    val_texts: np.ndarray,
    val_intents: np.ndarray,
    test_texts: np.ndarray,
    test_intents: np.ndarray,
    use_subset: bool,
    allowed_intents: List[str],
):
    if use_subset and not allowed_intents:
        allowed_intents = load_selected_intents_from_file(SELECTED_INTENTS_PATH)

    if not use_subset or not allowed_intents:
        return (
            train_texts,
            train_intents,
            val_texts,
            val_intents,
            test_texts,
            test_intents,
        )

    allowed_intents = set(allowed_intents)

    def filt(texts, intents):
        mask = np.array([label in allowed_intents for label in intents])
        return texts[mask], intents[mask]

    train_texts_f, train_intents_f = filt(train_texts, train_intents)
    val_texts_f, val_intents_f = filt(val_texts, val_intents)
    test_texts_f, test_intents_f = filt(test_texts, test_intents)

    print("\nUsing subset of intents:")
    print("  intents:", sorted(np.unique(train_intents_f)))
    print("  train size:", train_texts_f.shape[0])
    print("  val size:  ", val_texts_f.shape[0])
    print("  test size: ", test_texts_f.shape[0])

    return (
        train_texts_f,
        train_intents_f,
        val_texts_f,
        val_intents_f,
        test_texts_f,
        test_intents_f,
    )


def build_text_vectorizer(train_texts: np.ndarray) -> layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=MAX_LEN,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    ds = tf.data.Dataset.from_tensor_slices(train_texts).batch(128)
    vectorizer.adapt(ds)
    return vectorizer


def build_intent_model(
    num_classes: int,
    vectorizer: layers.TextVectorization,
) -> keras.Model:
    text_input = keras.Input(shape=(1,), dtype=tf.string, name="text_input")
    x = vectorizer(text_input)
    x = layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=128,
        name="embedding",
    )(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(text_input, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_tf_datasets(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    test_texts,
    test_labels,
):
    train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)) \
        .shuffle(len(train_texts)) \
        .batch(BATCH_SIZE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_labels)) \
        .batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_texts, test_labels)) \
        .batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds


def plot_training_history(history, save_path="intent_training_history.png"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Intent Model Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Intent Model Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


def evaluate_and_report(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    y_test_encoded: np.ndarray,
    label_encoder: LabelEncoder,
):
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)

    target_names = label_encoder.classes_
    print("\nTest classification report:")
    print(classification_report(y_test_encoded, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Intent Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig("intent_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved to intent_confusion_matrix.png")


def main():
    print("=" * 60)
    print("Deep Learning Intent Classification - CLINC150 (data_full.json)")
    print("=" * 60)

    (
        train_texts,
        train_intents,
        val_texts,
        val_intents,
        test_texts,
        test_intents,
    ) = load_clinc_json(JSON_PATH)

    (
        train_texts,
        train_intents,
        val_texts,
        val_intents,
        test_texts,
        test_intents,
    ) = maybe_filter_subset(
        train_texts,
        train_intents,
        val_texts,
        val_intents,
        test_texts,
        test_intents,
        USE_SUBSET,
        SELECTED_INTENTS,
    )

    all_intents = np.concatenate([train_intents, val_intents, test_intents])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_intents)

    train_labels = label_encoder.transform(train_intents)
    val_labels = label_encoder.transform(val_intents)
    test_labels = label_encoder.transform(test_intents)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    print(f"\nNumber of classes: {num_classes}")
    print("Classes:", class_names)

    np.save(LABEL_ENCODER_PATH, label_encoder.classes_)
    print(f"Label encoder classes saved to {LABEL_ENCODER_PATH}")

    vectorizer = build_text_vectorizer(train_texts)

    model = build_intent_model(num_classes, vectorizer)
    model.summary()

    train_ds, val_ds, test_ds = make_tf_datasets(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            "best_intent_model_clinc.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(history)

    print("\nEvaluating on test set...")
    evaluate_and_report(model, test_ds, test_labels, label_encoder)

    model.save(MODEL_PATH)
    print(f"\nFinal intent model saved to: {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("Intent model training complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
