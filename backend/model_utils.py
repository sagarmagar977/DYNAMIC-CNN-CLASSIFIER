import os
import json
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import Callback
from PIL import Image
import io
import matplotlib.cm as cm
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
import cv2
import threading

# Constants
IMG_SIZE = (224, 224)
CLASSES = ["messi", "yamal", "lewandowski"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_cnn.keras")
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_status.json")
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")

# Thread locks and in-memory caches
_model_lock = threading.Lock()
_cascade_lock = threading.Lock()
_model_cache = None
_frontal_cascade = None
_profile_cascade = None

def get_cascades():
    """Lazy-loads and caches the cascade classifiers in memory using a double-checked locking pattern."""
    global _frontal_cascade, _profile_cascade
    if _frontal_cascade is None or _profile_cascade is None:
        with _cascade_lock:
            if _frontal_cascade is None or _profile_cascade is None:
                _frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                _profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    return _frontal_cascade, _profile_cascade

def auto_detect_and_crop_face(pil_image, strict=False):
    """Detects a face in a PIL Image using OpenCV's Haar Cascades (both frontal and profile, left & right),
    crops it precisely with a center-seeking square bounding box to avoid aspect ratio distortion, and returns it.
    If no face is found:
      - If strict=True, returns None.
      - If strict=False, returns the original image.
    """
    try:
        # 1. Force conversion to RGB to handle Grayscale or RGBA images
        rgb_image = pil_image.convert('RGB')
        img_np = np.array(rgb_image)
        img_h, img_w, _ = img_np.shape
        
        # 2. Retrieve cached thread-safe cascade classifiers
        frontal_cascade, profile_cascade = get_cascades()
        
        # Convert to grayscale for Haar detector
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Try frontal detection
        faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(25, 25))
        
        # Try profile face detection (looks to the right by default)
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(25, 25))
            
        # Try profile face detection looking left by horizontally flipping the grayscale frame
        if len(faces) == 0:
            gray_flipped = cv2.flip(gray, 1)
            faces_flipped = profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.05, minNeighbors=5, minSize=(25, 25))
            if len(faces_flipped) > 0:
                # Map coordinates of the flipped frame back to original non-flipped coordinates
                faces = []
                for (xf, yf, wf, hf) in faces_flipped:
                    x_orig = img_w - xf - wf
                    faces.append((x_orig, yf, wf, hf))
            
        if len(faces) > 0:
            # Sort by face size descending and choose the largest face region
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            
            # 3. Center-seeking square bounding box mechanism to avoid aspect ratio warping
            cx = x + w / 2
            cy = y + h / 2
            
            # Symmetrically expand the bounding box size by 30%
            side = int(max(w, h) * 1.30)
            
            # Ensure the square size does not exceed the smallest image dimension
            side = min(side, img_w, img_h)
            
            # Symmetrically position coordinates
            x1 = int(cx - side / 2)
            y1 = int(cy - side / 2)
            
            # Boundary shift logic to keep crop square and strictly inside image boundaries
            if x1 < 0:
                x1 = 0
            elif x1 + side > img_w:
                x1 = img_w - side
                
            if y1 < 0:
                y1 = 0
            elif y1 + side > img_h:
                y1 = img_h - side
                
            cropped_np = img_np[y1 : y1 + side, x1 : x1 + side]
            return Image.fromarray(cropped_np)
    except Exception as e:
        print(f"Error during auto face-crop: {e}")
        
    return None if strict else pil_image

def get_model():
    """Retrieves the trained model from in-memory cache, or loads it from disk if available.
    Uses a double-checked locking pattern to prevent concurrency race conditions under heavy load.
    """
    global _model_cache
    if _model_cache is None:
        with _model_lock:
            if _model_cache is None:
                if os.path.exists(MODEL_PATH):
                    print(f"Loading trained model from disk: {MODEL_PATH}")
                    try:
                        _model_cache = load_model(MODEL_PATH)
                    except Exception as e:
                        print(f"Error loading model from disk: {e}")
                else:
                    print("No saved model found on disk.")
    return _model_cache

class RealTimeStatsCallback(Callback):
    """Callback to write training statistics to a JSON file at the end of each epoch."""
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        # Clear stats at start
        with open(STATUS_PATH, 'w') as f:
            json.dump({
                "status": "training",
                "current_epoch": 0,
                "total_epochs": total_epochs,
                "history": {
                    "loss": [], "accuracy": [],
                    "val_loss": [], "val_accuracy": []
                }
            }, f)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            # Read existing history
            if os.path.exists(STATUS_PATH):
                with open(STATUS_PATH, 'r') as f:
                    data = json.load(f)
            else:
                data = {"history": {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}}

            data["status"] = "training"
            data["current_epoch"] = epoch + 1
            data["total_epochs"] = self.total_epochs
            
            # Append new metrics
            data["history"]["loss"].append(float(logs.get('loss', 0.0)))
            data["history"]["accuracy"].append(float(logs.get('accuracy', 0.0)))
            data["history"]["val_loss"].append(float(logs.get('val_loss', 0.0)))
            data["history"]["val_accuracy"].append(float(logs.get('val_accuracy', 0.0)))

            with open(STATUS_PATH, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error writing training status: {e}")

def create_model():
    """Builds a MobileNetV2-based transfer learning model for Barcelona footballers classification."""
    # Load pre-trained MobileNetV2 base (excluding classification head)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False, #removing the final classification layer of the MobileNetV2 and kwwping only feature-recognition part
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze pre-trained layers
    model = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
        base_model,
        GlobalAveragePooling2D(name='global_pooling'),
        Dense(64, activation='relu', name='dense_1'),
        Dropout(0.5, name='dropout'),
        Dense(len(CLASSES), activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_dataset_data():
    """Loads images from dataset directory, pre-processes them for MobileNetV2, and returns datasets and labels."""
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_folder):
            continue
            
        for filename in os.listdir(class_folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            filepath = os.path.join(class_folder, filename)
            try:
                # Load image, resize, and convert to numpy array
                img = load_img(filepath, target_size=IMG_SIZE)
                img_arr = img_to_array(img)
                
                # Preprocess for MobileNetV2 (scales pixels to [-1, 1])
                img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
                images.append(img_arr)
                
                # Make one-hot encoded label representation
                label = np.zeros(len(CLASSES))
                label[class_idx] = 1.0
                labels.append(label)
            except Exception as e:
                print(f"Skipping corrupted image {filepath}: {e}")
                
    if len(images) == 0:
        return None, None
        
    return np.array(images, dtype='float32'), np.array(labels, dtype='float32')

def train_cnn_model(epochs=15, batch_size=4):
    """Loads the player dataset and trains the MobileNetV2 transfer model, logging stats in real-time."""
    global _model_cache
    
    # Write init status
    with open(STATUS_PATH, 'w') as f:
        json.dump({"status": "starting", "current_epoch": 0, "total_epochs": epochs, "history": {}}, f)
        
    X, y = load_dataset_data()
    if X is None or len(X) < 3:
        with open(STATUS_PATH, 'w') as f:
            json.dump({"status": "failed", "error": "Not enough images in dataset. Please add images first."}, f)
        return False
        
    # Split into train and validation (80-20 split)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    val_split = int(0.2 * len(X))
    if val_split < 1:
        val_split = 1  # Ensure at least 1 validation image
        
    X_val, y_val = X[:val_split], y[:val_split]
    X_train, y_train = X[val_split:], y[val_split:]
    
    # Define and train model
    model = create_model()
    
    # Use real-time status callback to log metrics to JSON
    status_callback = RealTimeStatsCallback(epochs)
    
    print(f"Transfer training started: {len(X_train)} training samples, {len(X_val)} validation samples.")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[status_callback],
        verbose=1
    )
    
    # Save the model
    model.save(MODEL_PATH)
    
    # Update in-memory model cache
    _model_cache = model
    
    # Update final training status
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, 'r') as f:
            data = json.load(f)
    else:
        data = {"history": {}}
        
    data["status"] = "completed"
    with open(STATUS_PATH, 'w') as f:
        json.dump(data, f)
        
    print("Training finished successfully. Model saved and cached.")
    return True

def run_inference(image_bytes):
    """Runs prediction on custom image bytes. Returns list of dictionary predictions."""
    model = get_model()
    if model is None:
        return {"error": "Model not trained yet. Please train the transfer model first."}
        
    try:
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = auto_detect_and_crop_face(img, strict=False)
        img_resized = img.resize(IMG_SIZE)
        img_arr = img_to_array(img_resized)
        img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension (1, 128, 128, 3)
        
        # Predict
        predictions = model.predict(img_arr)[0]
        
        results = []
        for idx, score in enumerate(predictions):
            results.append({
                "class": CLASSES[idx],
                "confidence": float(score)
            })
            
        # Sort predictions by confidence desc
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        # Convert cropped & resized face image to base64 to show in the frontend
        buffered = io.BytesIO()
        img_resized.save(buffered, format="JPEG")
        cropped_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "predictions": results,
            "cropped_face": cropped_base64
        }
    except Exception as e:
        return {"error": f"Error running inference: {str(e)}"}

def extract_activation_maps(image_bytes):
    """Extracts output activations of the first convolutional layer (16 filters) of the base MobileNetV2 model."""
    model = get_model()
    if model is None:
        return None
        
    try:
        # Preprocess input image
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = auto_detect_and_crop_face(img, strict=False)
        img_resized = img.resize(IMG_SIZE)
        img_arr = img_to_array(img_resized)
        img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)
        
        # Isolate base MobileNetV2 model dynamically
        base_model = None
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower() or hasattr(layer, 'layers'):
                base_model = layer
                break
                
        if base_model is None:
            print("Could not isolate base model.")
            return None
        
        # Find the first Conv2D layer inside the base model
        first_conv = None
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                first_conv = layer
                break
                
        if first_conv is None:
            print("Could not find any Conv2D layer in base model.")
            return None
            
        # Create intermediate activation model
        activation_model = tf.keras.Model(inputs=base_model.input, outputs=first_conv.output)
        activations = activation_model.predict(img_arr)[0]  # Shape: (H, W, 32)
        
        # Export each of the first 16 filters as a base64 encoded image
        maps_base64 = []
        for i in range(16):
            feat_map = activations[:, :, i]
            
            # Normalize to 0-255 range
            f_min, f_max = feat_map.min(), feat_map.max()
            if f_max - f_min > 0:
                feat_map = 255 * (feat_map - f_min) / (f_max - f_min)
            feat_map = feat_map.astype('uint8')
            
            # Convert map to colorful image using colormap (Viridis gives a stunning scientific lookup)
            colored_map = cm.viridis(feat_map)  # returns RGBA [0.0, 1.0]
            colored_map = (colored_map[:, :, :3] * 255).astype('uint8') # drop alpha, scale to 0-255
            
            # Create PIL image
            pil_map = Image.fromarray(colored_map)
            pil_map_resized = pil_map.resize((128, 128), Image.Resampling.NEAREST)  # Upscale for better viewing
            
            # Save to memory as PNG
            buffer = io.BytesIO()
            pil_map_resized.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            maps_base64.append(img_str)
            
        return maps_base64
    except Exception as e:
        print(f"Error extracting activation maps: {e}")
        return None
