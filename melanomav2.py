import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization  # Para BatchNormalization
from tensorflow.keras.regularizers import l2  # Para regularización L2
from tensorflow.keras.callbacks import ReduceLROnPlateau  # Para ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- 1. Configuración de Rutas y Parámetros ---
# Asegúrate de que estas rutas apuntan a tus directorios de datos
# Sugerencia: Organiza tus datos en subcarpetas 'train', 'validation', 'test'
# y dentro de cada una, subcarpetas 'melanoma' y 'no_melanoma'.

# Ejemplo de estructura de directorios:
# dataset/
# ├── train/
# │   ├── melanoma/
# │   └── no_melanoma/
# ├── validation/
# │   ├── melanoma/
# │   └── no_melanoma/
# └── test/
#     ├── melanoma/
#     └── no_melanoma/

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image, ImageFile

# --- Configuración de seguridad para imágenes grandes ---
Image.MAX_IMAGE_PIXELS = None  # Desactiva la protección decompression bomb
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Permite cargar imágenes truncadas

# --- Función segura para cargar imágenes ---
def safe_load_image(image_path, target_size):
    """
    Carga una imagen de manera segura con manejo de errores.
    Devuelve un array numpy o None si hay error.
    """
    try:
        # Primero verifica si el archivo existe
        if not os.path.exists(image_path):
            print(f"Archivo no encontrado: {image_path}")
            return None
            
        # Intenta cargar la imagen
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        return img_array / 255.0  # Normalización
        
    except Exception as e:
        print(f"Error cargando imagen {image_path}: {str(e)}")
        return None

# --- Generador de datos personalizado ---
class SafeImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, target_size, batch_size, class_mode, classes, shuffle):
        self.directory = directory
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.classes = classes
        self.shuffle = shuffle
        self.class_indices = {class_name: idx for idx, class_name in enumerate(classes)}
        
        # Lista de todas las imágenes válidas
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(self.class_indices[class_name])
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        
        batch_images = []
        valid_labels = []
        
        for path, label in zip(batch_paths, batch_labels):
            img_array = safe_load_image(path, self.target_size)
            if img_array is not None:
                batch_images.append(img_array)
                valid_labels.append(label)
        
        if len(batch_images) == 0:
            # Devuelve un batch vacío como fallback
            return np.zeros((0, *self.target_size, 3)), np.zeros((0,))
            
        return np.array(batch_images), np.array(valid_labels)

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.image_paths, self.labels))
            np.random.shuffle(combined)
            self.image_paths, self.labels = zip(*combined)


DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 35 # Puedes ajustar esto, y EarlyStopping ayudará a parar a tiempo
NUM_CLASSES = 2 # Melanoma, No Melanoma
CLASS_NAMES = ['benign', 'cancer'] # Asegúrate que el orden coincida con el generador de imágenes

# --- 2. Preparación de Datos con ImageDataGenerator ---
# ImageDataGenerator ayuda a cargar imágenes y aplicar aumentos de datos en tiempo real.
# Esto es crucial para mejorar la robustez del modelo y evitar el sobreajuste.

# --- Generadores de Datos ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='reflect',
    brightness_range=[0.9, 1.1]
)

val_datagen = ImageDataGenerator(rescale=1./255) # Solo reescalar para validación
test_datagen = ImageDataGenerator(rescale=1./255)       # Solo reescalar para pruebas

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASS_NAMES,
    shuffle=True,
    seed=42  # Fija una semilla para reproducibilidad
)

# Asegúrate de que el validation_generator tenga shuffle=False
validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASS_NAMES,
    shuffle=False  # ¡Importante para validación!
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASS_NAMES,
    shuffle=False # ¡Importante! Para evaluar correctamente y asociar predicciones con etiquetas reales
)

# Verificar el mapeo de clases
print(f"Mapeo de clases del generador de entrenamiento: {train_generator.class_indices}")

# Construcción del Modelo CNN (Convolutional Neural Network)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# --- Optimizador y Callbacks ---
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.00001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_melanoma_classifier.keras', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
]

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', 
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall')])

model.summary()

# --- 6. Entrenamiento del Modelo ---
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)

# --- 7. Evaluación del Modelo ---
print("\n--- Evaluación en el conjunto de prueba ---")
results = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
loss = results[0]
accuracy = results[1]

print(f"\nMétricas completas de evaluación:")
print(f"Pérdida: {loss:.4f}")
print(f"Precisión: {accuracy:.4f}")
print(f"Precisión (precision): {results[2]:.4f}")  # Índice 2 corresponde a la métrica precision
print(f"Sensibilidad (recall): {results[3]:.4f}")  # Índice 3 corresponde a la métrica recall
print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

# --- 8. Visualización del Historial de Entrenamiento ---
plt.figure(figsize=(12, 5))

# Gráfico de Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Gráfico de Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.tight_layout()
plt.show()

# --- 9. Métricas Detalladas y Matriz de Confusión ---
print("\n--- Generando predicciones para el conjunto de prueba ---")
# Obtener las etiquetas verdaderas
test_labels = test_generator.classes

# Obtener las predicciones del modelo
predictions_prob = model.predict(test_generator)
predictions_class = (predictions_prob > 0.5).astype(int).flatten() # Convertir probabilidades a clases (0 o 1)

# Reporte de clasificación
print("\n--- Reporte de Clasificación ---")
#print(classification_report(test_labels, predictions_class, target_names=CLASS_NAMES))

# Matriz de Confusión
cm = confusion_matrix(test_labels, predictions_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()

# --- 10. Función para Predecir una Sola Imagen (Ejemplo) ---
def predict_melanoma(image_path, model, img_height, img_width, class_names):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Crear un batch con una sola imagen
    img_array = img_array / 255.0 # Normalizar igual que en el entrenamiento

    prediction = model.predict(img_array)
    predicted_class_index = (prediction > 0.5).astype(int)[0][0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[0][0] if predicted_class_index == 1 else (1 - prediction[0][0])

    print(f"\nImagen: {image_path}")
    print(f"Predicción: {predicted_class_name}")
    print(f"Confianza: {confidence:.2f}")

    plt.imshow(img)
    plt.title(f"Predicción: {predicted_class_name} (Confianza: {confidence:.2f})")
    plt.axis('off')
    plt.show()

# --- Ejemplo de uso de la función de predicción ---
# Asegúrate de tener una imagen de prueba en tu sistema para probar esto
try:
    sample_image_path = r'D:\Documentos\ProtocoloDoctorado2026a\skinCancerDetection_p01\data\test\cancer\ISIC_0024332.jpg' # ¡CANCER!
    predict_melanoma(sample_image_path, model, IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES)
    sample_image_path = r'D:\Documentos\ProtocoloDoctorado2026a\skinCancerDetection_p01\data\valid\benign\ISIC_0029823.jpg' # ¡BENIGNO!
    predict_melanoma(sample_image_path, model, IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES)
except FileNotFoundError:
    print("\nAdvertencia: No se encontró la imagen de ejemplo para la predicción individual. Por favor, actualiza la ruta.")