import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
DATASET_PATH = "brain_tumor_dataset"

# Labels:
# 0 = No Tumor
# 1 = Tumor
images = []
labels = []

for folder, label in [("no", 0), ("yes", 1)]:

    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Folder not found: {folder_path}. "
            "Check your dataset path and folder names."
        )

    for filename in os.listdir(folder_path):

        image_path = os.path.join(folder_path, filename)

        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Skip files that cannot be read
        if image is None:
            print("Skipping unreadable file:", image_path)
            continue

        # Resize image to 64 x 64
        image = cv2.resize(image, (64, 64))

        # Flatten image into 4096 features
        image = image.flatten()
        images.append(image)
        labels.append(label)

X = np.array(images, dtype=np.float32)
y = np.array(labels)
print("Dataset loaded successfully")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("No Tumor images:", np.sum(y == 0))
print("Tumor images:", np.sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

print("\nTraining SVM model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n========== SVM RESULTS ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Tumor", "Tumor"],
        zero_division=0
    )
)
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Tumor", "Tumor"]
)

display.plot()
plt.title("SVM Confusion Matrix")
plt.show()

num_images = min(6, len(X_test))
X_test_images = X_test.copy()
X_test_images = scaler.inverse_transform(X_test_images)

plt.figure(figsize=(12, 8))

for i in range(num_images):
    image = X_test_images[i].reshape(64, 64)

    actual_label = (
        "Tumor" if y_test[i] == 1 else "No Tumor"
    )

    predicted_label = (
        "Tumor" if y_pred[i] == 1 else "No Tumor"
    )

    # Mark correct or incorrect prediction
    result = "Correct" if y_test[i] == y_pred[i] else "Incorrect"

    plt.subplot(2, 3, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(
        f"Actual: {actual_label}\n"
        f"Predicted: {predicted_label}\n"
        f"{result}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\nProgram completed successfully!")