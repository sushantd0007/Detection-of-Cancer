import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Load Genomic Data
genomic_file_path = 'Fake_Genomic_Data.csv'
genomic_data = pd.read_csv(genomic_file_path)
genomic_features = genomic_data.drop('Label', axis=1).values
labels = genomic_data['Label'].values.reshape(-1, 1)

# Load Clinical Data
clinical_file_path = 'Generated_Clinical_Dataset.csv'
clinical_data = pd.read_csv(clinical_file_path)

# Identify categorical and numeric columns
categorical_columns = ['Gender', 'Smoking_Status', 'Tumor_Stage']  # Replace with your actual categorical column names
numeric_columns = [col for col in clinical_data.columns if col not in categorical_columns + ['Label']]

# Apply OneHotEncoding to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ]
)

# Transform clinical features
clinical_features = preprocessor.fit_transform(clinical_data.drop('Label', axis=1))

# Load Image Data
image_file_path = 'Fake_Image_Data.npy'
image_data = np.load(image_file_path)

# Split the data into training and testing sets
genomic_train, genomic_test, clinical_train, clinical_test, image_train, image_test, y_train, y_test = train_test_split(
    genomic_features, clinical_features, image_data, labels, test_size=0.2, random_state=42
)

# Scale Genomic and Clinical Data
scaler_genomic = StandardScaler()
scaler_clinical = StandardScaler()

genomic_train = scaler_genomic.fit_transform(genomic_train)
genomic_test = scaler_genomic.transform(genomic_test)
clinical_train = scaler_clinical.fit_transform(clinical_train)
clinical_test = scaler_clinical.transform(clinical_test)

# Define Multimodal Model
def build_multimodal_model():
    # Genomic data branch
    input_genomic = tf.keras.Input(shape=(genomic_train.shape[1],))
    x1 = tf.keras.layers.Dense(128, activation="relu")(input_genomic)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dropout(0.3)(x1)

    # Clinical data branch
    input_clinical = tf.keras.Input(shape=(clinical_train.shape[1],))
    x2 = tf.keras.layers.Dense(64, activation="relu")(input_clinical)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)

    # Image data branch
    input_image = tf.keras.Input(shape=image_train.shape[1:])
    x3 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(input_image)
    x3 = tf.keras.layers.MaxPooling2D((2, 2))(x3)
    x3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x3)
    x3 = tf.keras.layers.MaxPooling2D((2, 2))(x3)
    x3 = tf.keras.layers.Flatten()(x3)
    x3 = tf.keras.layers.Dense(64, activation="relu")(x3)

    # Combine branches
    combined = tf.keras.layers.concatenate([x1, x2, x3])
    x = tf.keras.layers.Dense(64, activation="relu")(combined)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Compile model
    model = tf.keras.Model(inputs=[input_genomic, input_clinical, input_image], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Build and Train the Model
model = build_multimodal_model()
model.summary()

history = model.fit(
    [genomic_train, clinical_train, image_train],
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32
)

# Evaluate the Model
loss, accuracy = model.evaluate([genomic_test, clinical_test, image_test], y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Generate Evaluation Metrics
y_pred = (model.predict([genomic_test, clinical_test, image_test]) > 0.5).astype("int32")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and display ROC-AUC score
y_prob = model.predict([genomic_test, clinical_test, image_test])
roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC Score: {roc_auc:.2f}")

# Simple Output for Better Understanding
num_correct = np.sum(y_pred == y_test)
total_samples = len(y_test)
print(f"\nSimpler Output: Correct Predictions: {num_correct}/{total_samples} ({(num_correct/total_samples)*100:.2f}% accuracy)")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.close()

print("\nPlots saved as 'accuracy_plot.png' and 'loss_plot.png'")

# Save the Model
model.save('multimodal_cancer_model.keras')
print("\nModel saved as 'multimodal_cancer_model.keras'")
