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
import h5py
from keras.src.models.functional import Functional
import io
import matplotlib.cm as cm
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
import cv2
import threading

# Constants
IMG_SIZE = (112, 112)
CLASSES = ["messi", "yamal", "lewandowski"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_cnn.keras")
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_status.json")
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_templates.json")

import urllib.request
import shutil

# Thread locks and in-memory caches
_model_lock = threading.Lock()
_cascade_lock = threading.Lock()
_model_cache = None
_dnn_net = None
_master_templates = {}

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
_active_session_id = None

def get_active_session_details():
    """Finds the active session ID, classes, and paths.
    If no sessions exist or none are active, initializes default barca_players.
    """
    global _active_session_id
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    
    # 1. If we have it in memory and it's valid, return it
    if _active_session_id is not None:
        metadata_path = os.path.join(SESSIONS_DIR, _active_session_id, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                return _active_session_id, meta.get("classes", []), os.path.join(SESSIONS_DIR, _active_session_id)
            except Exception:
                pass
                
    # 2. Scan sessions directory for active session
    active_id = None
    fallback_id = None
    
    if os.path.exists(SESSIONS_DIR):
        for name in sorted(os.listdir(SESSIONS_DIR)):
            sess_path = os.path.join(SESSIONS_DIR, name)
            if os.path.isdir(sess_path):
                meta_path = os.path.join(sess_path, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        if meta.get("is_active", False):
                            active_id = name
                            break
                        if fallback_id is None:
                            fallback_id = name
                    except Exception:
                        pass
                        
    # If no active session found, use fallback or barca_players
    if active_id is None:
        active_id = fallback_id if fallback_id is not None else "barca_players"
        
    session_dir = os.path.join(SESSIONS_DIR, active_id)
    os.makedirs(session_dir, exist_ok=True)
    
    metadata_path = os.path.join(session_dir, "metadata.json")
    
    # Create default metadata if missing
    if not os.path.exists(metadata_path):
        classes = ["messi", "yamal", "lewandowski"]
        meta = {
            "id": active_id,
            "display_name": "Barcelona Players" if active_id == "barca_players" else active_id.replace("_", " ").title(),
            "classes": classes,
            "created_at": "2026-07-13T09:40:00Z",
            "status": "untrained",
            "is_active": True,
            "history": {
                "loss": [],
                "accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
        }
        with open(metadata_path, 'w') as f:
            json.dump(meta, f)
    else:
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            # Ensure it is marked active
            if not meta.get("is_active", False):
                meta["is_active"] = True
                with open(metadata_path, 'w') as f:
                    json.dump(meta, f)
        except Exception:
            classes = ["messi", "yamal", "lewandowski"]
            meta = {"id": active_id, "classes": classes}
            
    _active_session_id = active_id
    return active_id, meta.get("classes", []), session_dir

def get_active_classes():
    _, classes, _ = get_active_session_details()
    return classes

def get_active_dataset_dir():
    _, _, session_dir = get_active_session_details()
    dataset_dir = os.path.join(session_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir

def get_active_templates_path():
    _, _, session_dir = get_active_session_details()
    return os.path.join(session_dir, "master_templates.json")

def get_active_metadata_path():
    _, _, session_dir = get_active_session_details()
    return os.path.join(session_dir, "metadata.json")

def update_active_session_metadata(updates: dict):
    _, _, session_dir = get_active_session_details()
    metadata_path = os.path.join(session_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        meta = {}
        
    meta.update(updates)
    with open(metadata_path, 'w') as f:
        json.dump(meta, f)
    return meta

def activate_session(session_id):
    global _active_session_id, _master_templates
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    target_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Session directory {session_id} does not exist.")
        
    # Deactivate all sessions
    for name in os.listdir(SESSIONS_DIR):
        sess_path = os.path.join(SESSIONS_DIR, name)
        if os.path.isdir(sess_path):
            meta_path = os.path.join(sess_path, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    meta["is_active"] = (name == session_id)
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f)
                except Exception as e:
                    print(f"Error setting active state: {e}")
                    
    _active_session_id = session_id
    
    # Reload active templates into memory
    templates_path = os.path.join(target_dir, "master_templates.json")
    if os.path.exists(templates_path):
        try:
            with open(templates_path, 'r') as f:
                _master_templates = json.load(f)
            print(f"Loaded master templates for activated session: {session_id}")
        except Exception as e:
            print(f"Warning: Could not load templates for session {session_id}: {e}")
            _master_templates = {}
    else:
        _master_templates = {}
        
    return True

def list_sessions():
    sessions = []
    if os.path.exists(SESSIONS_DIR):
        for name in sorted(os.listdir(SESSIONS_DIR)):
            sess_path = os.path.join(SESSIONS_DIR, name)
            if os.path.isdir(sess_path):
                meta_path = os.path.join(sess_path, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        sessions.append(meta)
                    except Exception:
                        pass
    return sessions

def get_session_details(session_id):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    metadata_path = os.path.join(session_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None

def auto_migrate_legacy_dataset():
    """Migrates the legacy dataset directory to the default barca_players session directory on startup."""
    legacy_dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    
    target_session_dir = os.path.join(SESSIONS_DIR, "barca_players")
    target_dataset_dir = os.path.join(target_session_dir, "dataset")
    
    # Check if there is data to migrate
    if os.path.exists(legacy_dataset_dir) and not os.path.exists(target_dataset_dir):
        # Check if legacy directory has any player folders with images
        has_data = False
        for folder in ["messi", "yamal", "lewandowski"]:
            folder_path = os.path.join(legacy_dataset_dir, folder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(files) > 0:
                    has_data = True
                    break
                    
        if has_data:
            print(f"Auto-migrating legacy dataset from {legacy_dataset_dir} to {target_dataset_dir}...")
            try:
                os.makedirs(target_session_dir, exist_ok=True)
                shutil.copytree(legacy_dataset_dir, target_dataset_dir, dirs_exist_ok=True)
                
                # Copy existing master templates if present
                legacy_templates = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_templates.json")
                target_templates = os.path.join(target_session_dir, "master_templates.json")
                if os.path.exists(legacy_templates) and not os.path.exists(target_templates):
                    shutil.copy2(legacy_templates, target_templates)
                    
                # Migrate status/history
                legacy_status = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_status.json")
                history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
                status_str = "completed"
                if os.path.exists(legacy_status):
                    try:
                        with open(legacy_status, 'r') as f:
                            stat_data = json.load(f)
                        history = stat_data.get("history", history)
                        status_str = stat_data.get("status", "completed")
                    except Exception:
                        pass
                
                # Create metadata.json
                meta = {
                    "id": "barca_players",
                    "display_name": "Barcelona Players",
                    "classes": ["messi", "yamal", "lewandowski"],
                    "created_at": "2026-07-13T09:40:00Z",
                    "status": status_str,
                    "is_active": True,
                    "history": history
                }
                with open(os.path.join(target_session_dir, "metadata.json"), 'w') as f:
                    json.dump(meta, f)
                    
                print("Auto-migration of legacy dataset completed successfully.")
            except Exception as e:
                print(f"Error during legacy dataset auto-migration: {e}")

# Load cached master templates from disk if available for active session
active_id, active_classes, session_dir = get_active_session_details()
active_templates_path = os.path.join(session_dir, "master_templates.json")
if os.path.exists(active_templates_path):
    try:
        with open(active_templates_path, 'r') as f:
            _master_templates = json.load(f)
        print(f"Successfully loaded cached master templates from disk for active session: {active_id}.")
    except Exception as e:
        print(f"Warning: Could not load cached master templates: {e}")

# Paths for the DNN Face Detector
DNN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "local_temp", "dnn_face"))
PROTO_PATH = os.path.join(DNN_DIR, "deploy.prototxt")
MODEL_WEIGHTS_PATH = os.path.join(DNN_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

def get_dnn_net():
    """Lazy-loads and caches the OpenCV DNN face detector network.
    Automatically downloads the prototxt and model weights if they are not present.
    Returns None if downloading or loading fails.
    """
    global _dnn_net
    if _dnn_net is None:
        with _cascade_lock:
            if _dnn_net is None:
                try:
                    os.makedirs(DNN_DIR, exist_ok=True)
                    # Download files if missing
                    if not os.path.exists(PROTO_PATH):
                        print("Downloading OpenCV DNN Face Detector config (deploy.prototxt)...")
                        urllib.request.urlretrieve(
                            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                            PROTO_PATH
                        )
                    if not os.path.exists(MODEL_WEIGHTS_PATH):
                        print("Downloading OpenCV DNN Face Detector weights (10.6MB)...")
                        urllib.request.urlretrieve(
                            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                            MODEL_WEIGHTS_PATH
                        )
                    
                    _dnn_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_WEIGHTS_PATH)
                    print("OpenCV DNN Face Detector successfully loaded.")
                except Exception as e:
                    print(f"Warning: Could not load DNN face detector. Error: {e}")
                    _dnn_net = None
    return _dnn_net

def auto_detect_and_crop_face(pil_image, strict=False):
    """Detects a face in a PIL Image using OpenCV's DNN Face Detector,
    crops it precisely with a center-seeking square bounding box, and returns it.
    Returns None if no face is detected and strict=True.
    """
    try:
        # 1. Force conversion to RGB
        rgb_image = pil_image.convert('RGB')
        img_np = np.array(rgb_image)
        img_h, img_w, _ = img_np.shape
        
        # 2. Try the advanced DNN Face Detector
        dnn_net = get_dnn_net()
        faces = []
        
        if dnn_net is not None:
            # Preprocess image for DNN (size: 300x300, scale: 1.0, mean subtraction values: 104.0, 177.0, 123.0)
            # We set swapRB=True because img_np is RGB (from PIL), but the Caffe ResNet model expects BGR.
            blob = cv2.dnn.blobFromImage(cv2.resize(img_np, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True)
            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            
            # Parse detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4: # Lower threshold to 0.4 for better sensitivity on small/distant faces
                    box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    # Ensure coordinates are within boundary limits
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        faces.append((x1, y1, x2 - x1, y2 - y1))
                        
        if len(faces) > 0:
            # Sort by face size descending and choose the largest region
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            
            # 3. Center-seeking square bounding box mechanism
            cx = x + w / 2
            cy = y + h / 2
            
            # Expand by 30% to capture full head context comfortably
            side = int(max(w, h) * 1.30)
            side = min(side, img_w, img_h)
            
            x1 = int(cx - side / 2)
            y1 = int(cy - side / 2)
            
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

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobilefacenet_weights.h5")

def download_weights_if_missing():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Pre-trained MobileFaceNet weights file is missing at: {WEIGHTS_PATH}. "
            "The system cannot function without these pre-trained weights."
        )

def get_model():
    """Retrieves the trained model from in-memory cache, or loads it from disk if available."""
    global _model_cache
    if _model_cache is None:
        with _model_lock:
            if _model_cache is None:
                download_weights_if_missing()
                loaded_ok = False
                if os.path.exists(MODEL_PATH):
                    print(f"Loading trained model from disk: {MODEL_PATH}")
                    try:
                        temp_model = load_model(MODEL_PATH, safe_mode=False)
                        # Verify that the model output matches our 512-dimensional embedding size
                        if temp_model.output_shape[-1] == 512:
                            _model_cache = temp_model
                            loaded_ok = True
                            print("Loaded valid 512-D feature extractor model from disk.")
                        else:
                            print(f"Discarding saved model because of output shape mismatch: {temp_model.output_shape}")
                            try:
                                os.remove(MODEL_PATH)
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Error loading model from disk: {e}")
                
                if not loaded_ok:
                    print("Creating new MobileFaceNet feature embedding extractor...")
                    _model_cache = create_model()
                    try:
                        _model_cache.save(MODEL_PATH)
                        print(f"Saved rebuilt 512-D feature extractor model to {MODEL_PATH}")
                    except Exception as e:
                        print(f"Warning: Could not save model to disk: {e}")
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

def conv_block(x, filters, kernel_size, strides, padding='same', name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name + '_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=name + '_bn')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_prelu')(x)
    return x

def dw_conv_block(x, filters, kernel_size, strides, padding='same', name=None):
    x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False, name=name + '_dwconv')(x)
    x = tf.keras.layers.BatchNormalization(name=name + '_bn')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_prelu')(x)
    return x

def bottleneck(x, out_filters, strides, expansion, residual=True, name=None):
    in_filters = x.shape[-1]
    # 1. Expand
    expanded_filters = in_filters * expansion
    y = tf.keras.layers.Conv2D(expanded_filters, (1, 1), strides=1, padding='same', use_bias=False, name=name + '_expand')(x)
    y = tf.keras.layers.BatchNormalization(name=name + '_expand_bn')(y)
    y = tf.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_expand_prelu')(y)
    
    # 2. Depthwise
    y = tf.keras.layers.DepthwiseConv2D((3, 3), strides=strides, padding='same', use_bias=False, name=name + '_dw')(y)
    y = tf.keras.layers.BatchNormalization(name=name + '_dw_bn')(y)
    y = tf.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_dw_prelu')(y)
    
    # 3. Project
    y = tf.keras.layers.Conv2D(out_filters, (1, 1), strides=1, padding='same', use_bias=False, name=name + '_project')(y)
    y = tf.keras.layers.BatchNormalization(name=name + '_project_bn')(y)
    
    if residual and strides == 1 and in_filters == out_filters:
        y = tf.keras.layers.Add(name=name + '_add')([x, y])
    return y

def MobileFaceNet(input_shape=(112, 112, 3), embedding_size=512):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Conv1
    x = conv_block(inputs, 64, (3, 3), strides=2, name='conv1')
    
    # Conv2_dw
    x = dw_conv_block(x, 64, (3, 3), strides=1, name='conv2_dw')
    
    # Bottlenecks
    # Block 1
    x = bottleneck(x, 64, strides=2, expansion=2, residual=False, name='b1_1')
    x = bottleneck(x, 64, strides=1, expansion=2, residual=True, name='b1_2')
    x = bottleneck(x, 64, strides=1, expansion=2, residual=True, name='b1_3')
    x = bottleneck(x, 64, strides=1, expansion=2, residual=True, name='b1_4')
    x = bottleneck(x, 64, strides=1, expansion=2, residual=True, name='b1_5')
    
    # Block 2
    x = bottleneck(x, 128, strides=2, expansion=4, residual=False, name='b2_1')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b2_2')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b2_3')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b2_4')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b2_5')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b2_6')
    
    # Block 3
    x = bottleneck(x, 128, strides=2, expansion=4, residual=False, name='b3_1')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b3_2')
    
    # Conv3
    x = conv_block(x, 512, (1, 1), strides=1, name='conv3')
    
    # Linear GDConv (Global Depthwise Conv)
    x = tf.keras.layers.DepthwiseConv2D((7, 7), strides=1, padding='valid', use_bias=False, name='gdconv')(x)
    x = tf.keras.layers.BatchNormalization(name='gdconv_bn')(x)
    
    # Linear Conv2D to output embedding
    x = tf.keras.layers.Conv2D(embedding_size, (1, 1), strides=1, padding='valid', use_bias=False, name='embedding')(x)
    x = tf.keras.layers.BatchNormalization(name='embedding_bn')(x)
    
    x = tf.keras.layers.Flatten(name='flatten')(x)
    
    model = tf.keras.Model(inputs, x, name='MobileFaceNet')
    return model

def create_model():
    """Builds a MobileFaceNet-based feature embedding model using the loaded pre-trained weights."""
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Pre-trained MobileFaceNet weights file is missing at: {WEIGHTS_PATH}. "
            "The system cannot function without these pre-trained weights."
        )
        
    try:
        # Load the model configuration from the h5 attributes
        with h5py.File(WEIGHTS_PATH, 'r') as f:
            if 'model_config' not in f.attrs:
                raise KeyError(f"'model_config' attribute not found inside {WEIGHTS_PATH}.")
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            config = json.loads(config_str)
            
        # Clean SeparableConv2D layers config for Keras 3 compatibility
        layers = config['config']['layers']
        for layer in layers:
            if layer['class_name'] == 'SeparableConv2D':
                c = layer['config']
                for k in ['kernel_initializer', 'kernel_regularizer', 'kernel_constraint']:
                    if k in c:
                        del c[k]
                        
        # Load the base functional model directly from config
        base_model = Functional.from_config(config['config'])
        base_model.load_weights(WEIGHTS_PATH)
        print("Pre-trained MobileFaceNet base model loaded and weights bound successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Catastrophic failure loading or instantiating pre-trained model weights from {WEIGHTS_PATH}: {e}"
        )
        
    # Freeze all layers in the base backbone model
    for layer in base_model.layers:
        layer.trainable = False
        
    # Add a projection layer on top to yield a 512-dimensional linear feature embedding
    inputs = base_model.input
    x = base_model.output[0]
    outputs = tf.keras.layers.Dense(512, activation=None, name='embedding_projection_512')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='projected_MobileFaceNet')
    
    # Freeze the entire model
    for layer in model.layers:
        layer.trainable = False
        
    return model

def load_dataset_data():
    """Loads images from dataset directory, pre-processes them for MobileNetV2, and returns datasets and labels."""
    images = []
    labels = []
    
    active_classes = get_active_classes()
    active_dataset_dir = get_active_dataset_dir()
    
    for class_idx, class_name in enumerate(active_classes):
        class_folder = os.path.join(active_dataset_dir, class_name)
        if not os.path.exists(class_folder):
            continue
            
        for filename in os.listdir(class_folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            filepath = os.path.join(class_folder, filename)
            try:
                # Load image at full scale
                img = load_img(filepath)
                
                # Apply face detection and square cropping (falls back to original image if no face detected)
                img_cropped = auto_detect_and_crop_face(img, strict=False)
                img_resized = img_cropped.resize(IMG_SIZE)
                
                img_arr = img_to_array(img_resized)
                
                # Preprocess for MobileNetV2 (scales pixels to [-1, 1])
                img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
                images.append(img_arr)
                
                # Make one-hot encoded label representation
                label = np.zeros(len(active_classes))
                label[class_idx] = 1.0
                labels.append(label)
            except Exception as e:
                print(f"Skipping corrupted image {filepath}: {e}")
                
    if len(images) == 0:
        return None, None
        
    return np.array(images, dtype='float32'), np.array(labels, dtype='float32')

def l2_normalize(x, axis=-1, epsilon=1e-10):
    """L2 Normalization helper for feature vectors."""
    return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

def compute_master_templates():
    """Generates and averages 512-D vectors for training images to create Master Vector Templates."""
    global _master_templates
    model = get_model()
    if model is None:
        print("Warning: Model not available. Cannot compute master templates.")
        return
        
    print("Computing Master Vector Templates from dataset...")
    templates = {}
    
    active_classes = get_active_classes()
    active_dataset_dir = get_active_dataset_dir()
    active_templates_path = get_active_templates_path()
    
    for class_name in active_classes:
        class_folder = os.path.join(active_dataset_dir, class_name)
        if not os.path.exists(class_folder):
            continue
            
        embeddings = []
        for filename in os.listdir(class_folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            filepath = os.path.join(class_folder, filename)
            try:
                img = load_img(filepath)
                img_cropped = auto_detect_and_crop_face(img, strict=False)
                img_resized = img_cropped.resize(IMG_SIZE)
                img_arr = img_to_array(img_resized)
                img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
                img_arr = np.expand_dims(img_arr, axis=0)
                
                # Extract 512-D vector signature
                emb = model.predict(img_arr, verbose=0)[0]
                emb = l2_normalize(emb)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error extracting embedding for {filepath}: {e}")
                
        if len(embeddings) > 0:
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = l2_normalize(avg_emb)
            templates[class_name] = avg_emb.tolist()
            print(f"Generated Master Vector Template for {class_name} ({len(embeddings)} source frames).")
            
    _master_templates = templates
    
    try:
        with open(active_templates_path, 'w') as f:
            json.dump(_master_templates, f)
    except Exception as e:
        print(f"Error saving master templates to disk: {e}")

def train_cnn_model(epochs=5, batch_size=4):
    """Loads dataset images, extracts 512-D vectors, and runs Stratified K-Fold cross-validation
    with `epochs` folds. Logs metrics to training_status.json for live graphing,
    then computes and caches the final 100%-average templates.
    """
    global _master_templates
    folds = max(2, int(epochs)) # Ensure folds >= 2
    
    active_classes = get_active_classes()
    active_dataset_dir = get_active_dataset_dir()
    
    # Write starting status
    with open(STATUS_PATH, 'w') as f:
        json.dump({"status": "starting", "current_epoch": 0, "total_epochs": folds, "history": {}}, f)
        
    update_active_session_metadata({
        "status": "starting",
        "history": {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    })
        
    try:
        from sklearn.model_selection import StratifiedKFold
        import time
        
        # 1. Load all images and extract their 512-D vectors
        print("Extracting embedding vectors for the entire dataset...")
        all_vectors = []
        all_labels = []
        
        for class_idx, class_name in enumerate(active_classes):
            class_folder = os.path.join(active_dataset_dir, class_name)
            if not os.path.exists(class_folder):
                continue
                
            for filename in os.listdir(class_folder):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                filepath = os.path.join(class_folder, filename)
                try:
                    img = load_img(filepath)
                    img_cropped = auto_detect_and_crop_face(img, strict=False)
                    img_resized = img_cropped.resize(IMG_SIZE)
                    img_arr = img_to_array(img_resized)
                    img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
                    img_arr = np.expand_dims(img_arr, axis=0)
                    
                    # Run inference to get 512-D vector
                    model = get_model()
                    emb = model.predict(img_arr, verbose=0)[0]
                    emb = l2_normalize(emb)
                    
                    all_vectors.append(emb)
                    all_labels.append(class_idx)
                except Exception as e:
                    print(f"Skipping corrupted image {filepath}: {e}")
                    
        if len(all_vectors) < folds:
            raise ValueError(f"Not enough valid images in dataset ({len(all_vectors)}) to run {folds}-fold cross validation.")
            
        X = np.array(all_vectors, dtype='float32')
        y = np.array(all_labels, dtype='int32')
        
        # Initialize K-Fold
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        
        history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # 2. Run K-Fold validation loop
        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Processing Fold {fold_idx + 1}/{folds}...")
            
            # Split vectors
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Compute temporary master templates for this fold
            fold_templates = {}
            for class_idx, class_name in enumerate(active_classes):
                class_vectors = X_train[y_train == class_idx]
                if len(class_vectors) > 0:
                    avg_vector = np.mean(class_vectors, axis=0)
                    fold_templates[class_idx] = l2_normalize(avg_vector)
                else:
                    fold_templates[class_idx] = np.zeros(512)
                    
            # Evaluate metrics on Training Folds
            train_correct = 0
            train_loss_sum = 0.0
            for vec, label_idx in zip(X_train, y_train):
                # Calculate similarities
                similarities = {c_idx: np.dot(vec, temp_vec) for c_idx, temp_vec in fold_templates.items()}
                pred_label = max(similarities, key=similarities.get)
                if pred_label == label_idx:
                    train_correct += 1
                train_loss_sum += (1.0 - similarities[label_idx])
                
            train_acc = train_correct / len(X_train)
            train_loss = train_loss_sum / len(X_train)
            
            # Evaluate metrics on Validation Fold
            val_correct = 0
            val_loss_sum = 0.0
            for vec, label_idx in zip(X_test, y_test):
                similarities = {c_idx: np.dot(vec, temp_vec) for c_idx, temp_vec in fold_templates.items()}
                pred_label = max(similarities, key=similarities.get)
                if pred_label == label_idx:
                    val_correct += 1
                val_loss_sum += (1.0 - similarities[label_idx])
                
            val_acc = val_correct / len(X_test)
            val_loss = val_loss_sum / len(X_test)
            
            # Record metrics
            history["loss"].append(float(train_loss))
            history["accuracy"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_accuracy"].append(float(val_acc))
            
            # Update live training status for this fold
            status_data = {
                "status": "training",
                "current_epoch": fold_idx + 1,
                "total_epochs": folds,
                "history": history
            }
            with open(STATUS_PATH, 'w') as f:
                json.dump(status_data, f)
                
            # Also update active session metadata.json
            update_active_session_metadata({
                "status": "training",
                "history": history
            })
                
            # Add a slight delay to allow the user to watch the folds update in the UI
            time.sleep(0.5)
            
        # 3. Compute final templates using 100% of dataset images
        compute_master_templates()
        
        # Save model file skeleton to disk so uvicorn startup check succeeds
        model = get_model()
        if model is not None:
            model.save(MODEL_PATH)
            
        # Write final completed status
        status_data = {
            "status": "completed",
            "current_epoch": folds,
            "total_epochs": folds,
            "history": history
        }
        with open(STATUS_PATH, 'w') as f:
            json.dump(status_data, f)
            
        update_active_session_metadata({
            "status": "completed",
            "history": history
        })
            
        print(f"Stratified {folds}-Fold cross validation completed successfully.")
        return True
    except Exception as e:
        print(f"Error during K-Fold validation: {e}")
        status_data = {"status": "failed", "error": str(e)}
        with open(STATUS_PATH, 'w') as f:
            json.dump(status_data, f)
        update_active_session_metadata({
            "status": "failed",
            "error": str(e)
        })
        return False

def run_inference(image_bytes):
    """Runs cosine similarity prediction on custom image bytes against Master Vector Templates."""
    global _master_templates
    model = get_model()
    if model is None:
        return {"error": "Model not loaded. Please ensure model exists."}
        
    active_id, active_classes, session_dir = get_active_session_details()
    active_templates_path = os.path.join(session_dir, "master_templates.json")
    
    global _loaded_templates_session_id
    if (not _master_templates) or (globals().get('_loaded_templates_session_id') != active_id):
        if os.path.exists(active_templates_path):
            try:
                with open(active_templates_path, 'r') as f:
                    _master_templates = json.load(f)
                _loaded_templates_session_id = active_id
                print(f"Loaded master templates for active session: {active_id}")
            except Exception as e:
                print(f"Warning: Could not load templates for session {active_id}: {e}")
                _master_templates = {}
        else:
            _master_templates = {}
            
    if not _master_templates:
        compute_master_templates()
        
    if not _master_templates:
        return {"error": "No training images available to compute templates. Please upload some training photos first."}
        
    try:
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        
        # Crop face strictly. If no face is found, return a clear error.
        img_cropped = auto_detect_and_crop_face(img, strict=True)
        if img_cropped is None:
            return {"error": "No face detected. Please upload a clear photo showing the player's face."}
            
        img_resized = img_cropped.resize(IMG_SIZE)
        img_arr = img_to_array(img_resized)
        img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)
        
        # Extract 512-D vector signature
        emb = model.predict(img_arr, verbose=0)[0]
        emb = l2_normalize(emb)
        
        # Compute Cosine Similarity against all Master Vector Templates
        results = []
        for class_name, template_emb in _master_templates.items():
            similarity = float(np.dot(emb, np.array(template_emb)))
            results.append({
                "class": class_name,
                "confidence": max(0.0, similarity)  # Present similarity score
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
        
        # Crop face strictly
        img_cropped = auto_detect_and_crop_face(img, strict=True)
        if img_cropped is None:
            return None
            
        img_resized = img_cropped.resize(IMG_SIZE)
        img_arr = img_to_array(img_resized)
        img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)
        
        # Isolate base model dynamically
        base_model = None
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower() or 'mobilefacenet' in layer.name.lower() or hasattr(layer, 'layers'):
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
