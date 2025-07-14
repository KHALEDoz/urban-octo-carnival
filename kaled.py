import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

dataset_path = r"/Users/khaled24/Downloads/DatasetCovid"

train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")
test_dir = os.path.join(dataset_path, "test")

results_dir = os.path.join(dataset_path, "training_results")
os.makedirs(results_dir, exist_ok=True)

num_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print(f"✅ Number of detected classes: {num_classes}")

image_size = (256, 256)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=3,
    zoom_range=0.05,
    width_shift_range=0.03,
    height_shift_range=0.03,
    brightness_range=[0.9, 1.1],
    horizontal_flip=False,
    vertical_flip=False
)

# فقط rescale للتحقق والاختبار
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# مولد بيانات التدريب مع التعديل
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# مولد بيانات التحقق بدون تعديل
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# مولد بيانات الاختبار بدون تعديل وبدون خلط
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# بناء موديل CNN تسلسلي
model = Sequential([
    Input(shape=(256, 256, 3)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(poo_size=(2, 2)),

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# حفظ أفضل موديل حسب دقة التحقق
checkpoint_path = os.path.join(results_dir, 'best_model.h5')
callbacks = [ModelCheckpoint(checkpoint_path, save_best_only=True)]

start_time = time.time()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,
    callbacks=callbacks
)

end_time = time.time()
training_time = end_time - start_time
print(f"⏱️ Training time: {training_time:.2f} seconds")

# رسم دقة التدريب والتحقق
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'))

# رسم خسارة التدريب والتحقق
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_dir, 'loss_plot.png'))

# حفظ ملخص التدريب في ملف نصي
with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")

print("✅ Training completed and results saved.")
