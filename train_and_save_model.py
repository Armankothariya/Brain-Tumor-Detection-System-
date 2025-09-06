import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -----------------------------
# 1. Paths
# -----------------------------
base_dir = r"C:\Users\user\AI_ML\INTERNSHIP\Brain tumor detection model"

# Final train/test folders (no "dataset" wrapper)
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Raw dataset folders (you must already have these)
classes = ["glioma", "meningioma", "no_tumor", "pituitary"]

if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    print("Splitting dataset into train/test...")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        class_dir = os.path.join(base_dir, cls)  # raw class folder
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        images = []
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    images.append(os.path.join(root, file))

        print(f"Found {len(images)} images for class {cls}")

        if len(images) == 0:
            continue

        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

        for img in train_imgs:
            shutil.copy(img, os.path.join(train_dir, cls, os.path.basename(img)))
        for img in test_imgs:
            shutil.copy(img, os.path.join(test_dir, cls, os.path.basename(img)))

# -----------------------------
# 2. Data Generators (with Augmentation)
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

print("Class indices:", train_generator.class_indices)

# -----------------------------
# 3. Build Transfer Learning Model
# -----------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(4, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------------
# 4. Callbacks
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)

# -----------------------------
# 5. Train Model
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop, reduce_lr]
)

# -----------------------------
# 6. Save Model (h5 format)
# -----------------------------
model.save("brain_tumor_model.h5")
print(" Model saved as brain_tumor_model.h5")

# -----------------------------
# 7. Evaluate
# -----------------------------
loss, acc = model.evaluate(val_generator)
print(f" Test Accuracy: {acc*100:.2f}%")
