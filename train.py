import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Check for GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ GPU available: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)  # Prevents memory allocation issues
else:
    print("❌ No GPU found. Training will run on CPU.")

# Define dataset paths
DATASET_PATH = "dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")

# Hyperparameters
IMG_SIZE = (384, 384)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = len(os.listdir(TRAIN_DIR))

# Data Augmentation
train_datagen = ImageDataGenerator(fill_mode="nearest")
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Load EfficientNetV2-S model
base_model = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Freeze base model

# Custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
checkpoint = ModelCheckpoint("efficientnetv2s_best.h5", save_best_only=True, monitor="val_accuracy", mode="max")
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the model on GPU
with tf.device("/GPU:0"):
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )

# Save final model
model.save("audi_classifier.h5")

print("✅ Training complete! Model saved as 'efficientnetv2s_final.h5'")
