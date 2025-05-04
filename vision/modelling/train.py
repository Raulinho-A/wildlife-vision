import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json

from model_zoo import build_baseline_model
from vision.plots import plot_training_history
from vision.config import DATALOADER, CHECKPOINTS_DIR

# Configuración
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30

DATA_DIR = str(DATALOADER)

# Cargar datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

model = build_baseline_model(IMG_HEIGHT, IMG_WIDTH, num_classes)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    ModelCheckpoint(str(CHECKPOINTS_DIR / 'best_model.keras'), monitor='val_loss', save_best_only=True)

]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(str(CHECKPOINTS_DIR / 'final_model.keras'))

val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

plot_training_history(history, title_prefix="Baseline Model")

history_path = CHECKPOINTS_DIR / 'baseline_model_history.json'
with open(history_path, 'w') as f:
    json.dump(history.history, f)

print(f"History guardado en {history_path}")