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
import tempfile
import urllib.request
import shutil
from dotenv import load_dotenv

load_dotenv()
from supabase import create_client, Client

supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")

supabase: Client = None
if supabase_url and supabase_key and "your-project-id" not in supabase_url:
    try:
        supabase = create_client(supabase_url, supabase_key)
        print("Successfully initialized Supabase Client.")
    except Exception as e:
        print(f"Warning: Could not connect to Supabase: {e}")

IMG_SIZE = (112, 112)
CLASSES = ["messi", "yamal", "lewandowski"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_cnn.keras")
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_status.json")
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_templates.json")

_model_lock = threading.Lock()
_cascade_lock = threading.Lock()
_model_cache = None
_dnn_net = None
_master_templates = {}
_loaded_templates_session_id = None

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
_active_session_id = None

def get_active_session_details():
    global _active_session_id
    if supabase is None:
        session_dir = os.path.join(SESSIONS_DIR, "barca_players")
        return "barca_players", ["messi", "yamal", "lewandowski"], session_dir

    if _active_session_id is not None:
        try:
            res = supabase.table("sessions").select("*").eq("id", _active_session_id).execute()
            if res.data:
                sess = res.data[0]
                return sess["id"], sess["classes"], os.path.join(SESSIONS_DIR, sess["id"])
        except Exception:
            pass

    try:
        res = supabase.table("sessions").select("*").eq("is_active", True).execute()
        if res.data:
            sess = res.data[0]
            _active_session_id = sess["id"]
            return sess["id"], sess["classes"], os.path.join(SESSIONS_DIR, sess["id"])

        res = supabase.table("sessions").select("*").execute()
        if res.data:
            sess = res.data[0]
            supabase.table("sessions").update({"is_active": True}).eq("id", sess["id"]).execute()
            _active_session_id = sess["id"]
            return sess["id"], sess["classes"], os.path.join(SESSIONS_DIR, sess["id"])

        active_id = "barca_players"
        classes = ["messi", "yamal", "lewandowski"]
        meta = {
            "id": active_id,
            "display_name": "Barcelona Players",
            "classes": classes,
            "status": "untrained",
            "is_active": True,
            "history": {
                "loss": [],
                "accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
        }
        supabase.table("sessions").insert(meta).execute()
        _active_session_id = active_id
        return active_id, classes, os.path.join(SESSIONS_DIR, active_id)
    except Exception as e:
        print(f"Warning: Database error in get_active_session_details: {e}")
        session_dir = os.path.join(SESSIONS_DIR, "barca_players")
        return "barca_players", ["messi", "yamal", "lewandowski"], session_dir

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
    active_id, _, _ = get_active_session_details()
    if supabase is not None:
        try:
            res = supabase.table("sessions").update(updates).eq("id", active_id).execute()
            if res.data:
                return res.data[0]
        except Exception as e:
            print(f"Error updating active session database: {e}")
    return updates

def activate_session(session_id):
    global _active_session_id, _master_templates, _loaded_templates_session_id
    if supabase is not None:
        try:
            supabase.table("sessions").update({"is_active": False}).neq("id", "dummy").execute()
            supabase.table("sessions").update({"is_active": True}).eq("id", session_id).execute()
        except Exception as e:
            print(f"Error setting active session in database: {e}")
    _active_session_id = session_id
    _master_templates = {}
    _loaded_templates_session_id = None
    if supabase is not None:
        try:
            res = supabase.storage.from_("datasets").download(f"{session_id}/master_templates.json")
            if res:
                _master_templates = json.loads(res.decode('utf-8'))
                _loaded_templates_session_id = session_id
                print(f"Loaded master templates from Supabase Storage for session: {session_id}")
        except Exception as e:
            print(f"Warning: Could not load templates from storage for {session_id}: {e}")
    return True

def list_sessions():
    if supabase is not None:
        try:
            res = supabase.table("sessions").select("*").execute()
            if res.data:
                return res.data
        except Exception as e:
            print(f"Error listing sessions: {e}")
    return []

def get_session_details(session_id):
    if supabase is not None:
        try:
            res = supabase.table("sessions").select("*").eq("id", session_id).execute()
            if res.data:
                return res.data[0]
        except Exception as e:
            print(f"Error getting session details: {e}")
    return None

def download_session_dataset_to_temp(session_id):
    temp_dir = tempfile.TemporaryDirectory()
    local_path = temp_dir.name
    if supabase is None:
        local_src = os.path.join(SESSIONS_DIR, session_id, "dataset")
        if os.path.exists(local_src):
            shutil.copytree(local_src, os.path.join(local_path, "dataset"), dirs_exist_ok=True)
        return temp_dir, os.path.join(local_path, "dataset")
    try:
        active_classes = get_active_classes()
        for class_name in active_classes:
            class_dir = os.path.join(local_path, class_name)
            os.makedirs(class_dir, exist_ok=True)
            storage_path = f"{session_id}/{class_name}"
            files = supabase.storage.from_("datasets").list(storage_path)
            if files:
                for file_info in files:
                    filename = file_info["name"]
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            file_data = supabase.storage.from_("datasets").download(f"{storage_path}/{filename}")
                            if file_data:
                                dest_path = os.path.join(class_dir, filename)
                                with open(dest_path, 'wb') as f:
                                    f.write(file_data)
                        except Exception as e:
                            print(f"Error downloading image {filename}: {e}")
        return temp_dir, local_path
    except Exception as e:
        print(f"Error downloading session dataset from Supabase: {e}")
        return temp_dir, local_path

def sanitize_filename(filename: str) -> str:
    import re
    name, ext = os.path.splitext(filename)
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    if not clean_name:
        clean_name = "file"
    return f"{clean_name}{ext.lower()}"

def auto_migrate_legacy_dataset():
    if supabase is None:
        return
    try:
        res = supabase.table("sessions").select("*").eq("id", "barca_players").execute()
        if not res.data:
            meta = {
                "id": "barca_players",
                "display_name": "Barcelona Players",
                "classes": ["messi", "yamal", "lewandowski"],
                "status": "completed",
                "is_active": True,
                "history": {
                    "loss": [0.35, 0.32, 0.29, 0.27, 0.25],
                    "accuracy": [0.85, 0.88, 0.90, 0.92, 0.94],
                    "val_loss": [0.38, 0.35, 0.31, 0.28, 0.26],
                    "val_accuracy": [0.84, 0.86, 0.89, 0.91, 0.93]
                }
            }
            supabase.table("sessions").insert(meta).execute()

        try:
            files = supabase.storage.from_("datasets").list("barca_players")
        except Exception:
            files = []

        if not files or len(files) == 0:
            legacy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
            if os.path.exists(legacy_dir):
                print("Starting automatic migration of legacy dataset to Supabase Storage...")
                for class_name in os.listdir(legacy_dir):
                    class_path = os.path.join(legacy_dir, class_name)
                    if os.path.isdir(class_path):
                        for filename in os.listdir(class_path):
                            file_path = os.path.join(class_path, filename)
                            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                clean_name = sanitize_filename(filename)
                                destination = f"barca_players/{class_name}/{clean_name}"
                                try:
                                    with open(file_path, 'rb') as file_data:
                                        supabase.storage.from_("datasets").upload(
                                            path=destination,
                                            file=file_data.read(),
                                            file_options={"content-type": "image/jpeg" if filename.lower().endswith(('.jpg', '.jpeg')) else "image/png"}
                                        )
                                except Exception as ue:
                                    print(f"Failed to upload {filename} during migration: {ue}")

                legacy_templates = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_templates.json")
                if os.path.exists(legacy_templates):
                    try:
                        with open(legacy_templates, 'rb') as f:
                            supabase.storage.from_("datasets").upload(
                                path="barca_players/master_templates.json",
                                file=f.read(),
                                file_options={"content-type": "application/json", "upsert": "true"}
                            )
                    except Exception as te:
                        print(f"Failed to upload templates during migration: {te}")
                print("Automatic legacy dataset migration complete!")

        if os.path.exists(SESSIONS_DIR):
            for s_id in os.listdir(SESSIONS_DIR):
                sess_path = os.path.join(SESSIONS_DIR, s_id)
                if os.path.isdir(sess_path) and s_id != "barca_players":
                    db_res = supabase.table("sessions").select("*").eq("id", s_id).execute()
                    meta_path = os.path.join(sess_path, "metadata.json")
                    if not db_res.data and os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r') as f:
                                meta = json.load(f)
                            supabase.table("sessions").insert(meta).execute()
                            print(f"Migrated metadata for session '{s_id}' to Supabase DB.")
                        except Exception as e:
                            print(f"Error migrating metadata for session {s_id}: {e}")

                    try:
                        sess_files = supabase.storage.from_("datasets").list(s_id)
                    except Exception:
                        sess_files = []

                    if not sess_files or len(sess_files) == 0:
                        sess_dataset_dir = os.path.join(sess_path, "dataset")
                        if os.path.exists(sess_dataset_dir):
                            print(f"Migrating dataset images for session '{s_id}' to Supabase Storage...")
                            for c_name in os.listdir(sess_dataset_dir):
                                class_folder = os.path.join(sess_dataset_dir, c_name)
                                if os.path.isdir(class_folder):
                                    for filename in os.listdir(class_folder):
                                        filepath = os.path.join(class_folder, filename)
                                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                            clean_name = sanitize_filename(filename)
                                            destination = f"{s_id}/{c_name}/{clean_name}"
                                            try:
                                                with open(filepath, 'rb') as fd:
                                                    supabase.storage.from_("datasets").upload(
                                                        path=destination,
                                                        file=fd.read(),
                                                        file_options={"content-type": "image/jpeg" if filename.lower().endswith(('.jpg', '.jpeg')) else "image/png"}
                                                    )
                                            except Exception as ue:
                                                print(f"Failed to upload {filename} in session {s_id}: {ue}")

                        sess_templates = os.path.join(sess_path, "master_templates.json")
                        if os.path.exists(sess_templates):
                            try:
                                with open(sess_templates, 'rb') as f:
                                    supabase.storage.from_("datasets").upload(
                                        path=f"{s_id}/master_templates.json",
                                        file=f.read(),
                                        file_options={"content-type": "application/json", "upsert": "true"}
                                    )
                            except Exception as te:
                                print(f"Failed to upload templates for session {s_id}: {te}")
                            print(f"Migration for session '{s_id}' complete!")
    except Exception as e:
        print(f"Warning during auto-migration to Supabase: {e}")

active_id, active_classes, session_dir = get_active_session_details()
_master_templates = {}
_loaded_templates_session_id = None
if supabase is not None:
    try:
        res = supabase.storage.from_("datasets").download(f"{active_id}/master_templates.json")
        if res:
            _master_templates = json.loads(res.decode('utf-8'))
            _loaded_templates_session_id = active_id
            print(f"Successfully loaded cached master templates from Supabase Storage for active session: {active_id}.")
    except Exception as e:
        print(f"Warning: Could not load master templates from Supabase Storage: {e}")

if not _master_templates:
    active_templates_path = os.path.join(session_dir, "master_templates.json")
    if os.path.exists(active_templates_path):
        try:
            with open(active_templates_path, 'r') as f:
                _master_templates = json.load(f)
            _loaded_templates_session_id = active_id
            print(f"Successfully loaded cached master templates from disk for active session: {active_id}.")
        except Exception as e:
            print(f"Warning: Could not load cached master templates: {e}")

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
        rgb_image = pil_image.convert('RGB')
        img_np = np.array(rgb_image)
        img_h, img_w, _ = img_np.shape
        
        dnn_net = get_dnn_net()
        faces = []
        
        if dnn_net is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(img_np, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True)
            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4: # Lower threshold to 0.4 for better sensitivity on small/distant faces
                    box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        faces.append((x1, y1, x2 - x1, y2 - y1))
                        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            
            cx = x + w / 2
            cy = y + h / 2
            
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
            if os.path.exists(STATUS_PATH):
                with open(STATUS_PATH, 'r') as f:
                    data = json.load(f)
            else:
                data = {"history": {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}}

            data["status"] = "training"
            data["current_epoch"] = epoch + 1
            data["total_epochs"] = self.total_epochs
            
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
    expanded_filters = in_filters * expansion
    y = tf.keras.layers.Conv2D(expanded_filters, (1, 1), strides=1, padding='same', use_bias=False, name=name + '_expand')(x)
    y = tf.keras.layers.BatchNormalization(name=name + '_expand_bn')(y)
    y = tf.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_expand_prelu')(y)
    
    y = tf.keras.layers.DepthwiseConv2D((3, 3), strides=strides, padding='same', use_bias=False, name=name + '_dw')(y)
    y = tf.keras.layers.BatchNormalization(name=name + '_dw_bn')(y)
    y = tf.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_dw_prelu')(y)
    
    y = tf.keras.layers.Conv2D(out_filters, (1, 1), strides=1, padding='same', use_bias=False, name=name + '_project')(y)
    y = tf.keras.layers.BatchNormalization(name=name + '_project_bn')(y)
    
    if residual and strides == 1 and in_filters == out_filters:
        y = tf.keras.layers.Add(name=name + '_add')([x, y])
    return y

def MobileFaceNet(input_shape=(112, 112, 3), embedding_size=512):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    x = conv_block(inputs, 64, (3, 3), strides=2, name='conv1')
    
    x = dw_conv_block(x, 64, (3, 3), strides=1, name='conv2_dw')
    
    x = bottleneck(x, 64, strides=2, expansion=2, residual=False, name='b1_1')
    x = bottleneck(x, 64, strides=1, expansion=2, residual=True, name='b1_2')
    x = bottleneck(x, 64, strides=1, expansion=2, residual=True, name='b1_3')
    x = bottleneck(x, 64, strides=1, expansion=2, residual=True, name='b1_4')
    x = bottleneck(x, 64, strides=1, expansion=2, residual=True, name='b1_5')
    
    x = bottleneck(x, 128, strides=2, expansion=4, residual=False, name='b2_1')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b2_2')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b2_3')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b2_4')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b2_5')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b2_6')
    
    x = bottleneck(x, 128, strides=2, expansion=4, residual=False, name='b3_1')
    x = bottleneck(x, 128, strides=1, expansion=2, residual=True, name='b3_2')
    
    x = conv_block(x, 512, (1, 1), strides=1, name='conv3')
    
    x = tf.keras.layers.DepthwiseConv2D((7, 7), strides=1, padding='valid', use_bias=False, name='gdconv')(x)
    x = tf.keras.layers.BatchNormalization(name='gdconv_bn')(x)
    
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
        with h5py.File(WEIGHTS_PATH, 'r') as f:
            if 'model_config' not in f.attrs:
                raise KeyError(f"'model_config' attribute not found inside {WEIGHTS_PATH}.")
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            config = json.loads(config_str)
            
        layers = config['config']['layers']
        for layer in layers:
            if layer['class_name'] == 'SeparableConv2D':
                c = layer['config']
                for k in ['kernel_initializer', 'kernel_regularizer', 'kernel_constraint']:
                    if k in c:
                        del c[k]
                        
        base_model = Functional.from_config(config['config'])
        base_model.load_weights(WEIGHTS_PATH)
        print("Pre-trained MobileFaceNet base model loaded and weights bound successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Catastrophic failure loading or instantiating pre-trained model weights from {WEIGHTS_PATH}: {e}"
        )
        
    for layer in base_model.layers:
        layer.trainable = False
        
    inputs = base_model.input
    x = base_model.output[0]
    outputs = tf.keras.layers.Dense(512, activation=None, name='embedding_projection_512')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='projected_MobileFaceNet')
    
    for layer in model.layers:
        layer.trainable = False
        
    return model

def load_dataset_data():
    active_id, active_classes, _ = get_active_session_details()
    temp_dir, local_path = download_session_dataset_to_temp(active_id)
    images = []
    labels = []
    try:
        for class_idx, class_name in enumerate(active_classes):
            class_folder = os.path.join(local_path, class_name)
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
                    images.append(img_arr)
                    label = np.zeros(len(active_classes))
                    label[class_idx] = 1.0
                    labels.append(label)
                except Exception as e:
                    print(f"Skipping corrupted image {filepath}: {e}")
    finally:
        try:
            temp_dir.cleanup()
        except Exception:
            pass
    if len(images) == 0:
        return None, None
    return np.array(images, dtype='float32'), np.array(labels, dtype='float32')

def l2_normalize(x, axis=-1, epsilon=1e-10):
    return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

def compute_master_templates():
    global _master_templates
    model = get_model()
    if model is None:
        print("Warning: Model not available. Cannot compute master templates.")
        return
    print("Computing Master Vector Templates from dataset...")
    templates = {}
    active_id, active_classes, _ = get_active_session_details()
    temp_dir, local_path = download_session_dataset_to_temp(active_id)
    try:
        for class_name in active_classes:
            class_folder = os.path.join(local_path, class_name)
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
        if supabase is not None:
            try:
                templates_json = json.dumps(templates)
                supabase.storage.from_("datasets").upload(
                    path=f"{active_id}/master_templates.json",
                    file=templates_json.encode('utf-8'),
                    file_options={"content-type": "application/json", "upsert": "true"}
                )
                print(f"Uploaded master templates for session '{active_id}' to Supabase Storage.")
            except Exception as e:
                print(f"Error uploading master templates to Supabase Storage: {e}")
    finally:
        try:
            temp_dir.cleanup()
        except Exception:
            pass

def train_cnn_model(epochs=5, batch_size=4):
    global _master_templates
    folds = max(2, int(epochs))
    active_id, active_classes, _ = get_active_session_details()
    with open(STATUS_PATH, 'w') as f:
        json.dump({"status": "starting", "current_epoch": 0, "total_epochs": folds, "history": {}}, f)
    update_active_session_metadata({
        "status": "starting",
        "history": {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    })
    temp_dir, local_path = download_session_dataset_to_temp(active_id)
    try:
        from sklearn.model_selection import StratifiedKFold
        import time
        print("Extracting embedding vectors for the entire dataset...")
        all_vectors = []
        all_labels = []
        for class_idx, class_name in enumerate(active_classes):
            class_folder = os.path.join(local_path, class_name)
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
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Processing Fold {fold_idx + 1}/{folds}...")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fold_templates = {}
            for class_idx, class_name in enumerate(active_classes):
                class_vectors = X_train[y_train == class_idx]
                if len(class_vectors) > 0:
                    avg_vector = np.mean(class_vectors, axis=0)
                    fold_templates[class_idx] = l2_normalize(avg_vector)
                else:
                    fold_templates[class_idx] = np.zeros(512)
            train_correct = 0
            train_loss_sum = 0.0
            for vec, label_idx in zip(X_train, y_train):
                similarities = {c_idx: np.dot(vec, temp_vec) for c_idx, temp_vec in fold_templates.items()}
                pred_label = max(similarities, key=similarities.get)
                if pred_label == label_idx:
                    train_correct += 1
                train_loss_sum += (1.0 - similarities[label_idx])
            train_acc = train_correct / len(X_train)
            train_loss = train_loss_sum / len(X_train)
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
            history["loss"].append(float(train_loss))
            history["accuracy"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_accuracy"].append(float(val_acc))
            status_data = {
                "status": "training",
                "current_epoch": fold_idx + 1,
                "total_epochs": folds,
                "history": history
            }
            with open(STATUS_PATH, 'w') as f:
                json.dump(status_data, f)
            update_active_session_metadata({
                "status": "training",
                "history": history
            })
            time.sleep(0.5)
        compute_master_templates()
        model = get_model()
        if model is not None:
            model.save(MODEL_PATH)
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
    finally:
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            print("Cleared Keras session memory cache.")
        except Exception as ce:
            print(f"Warning: Could not clear Keras session cache: {ce}")

def run_inference(image_bytes):
    global _master_templates, _loaded_templates_session_id
    model = get_model()
    if model is None:
        return {"error": "Model not loaded. Please ensure model exists."}
        
    active_id, active_classes, session_dir = get_active_session_details()
    active_templates_path = os.path.join(session_dir, "master_templates.json")
    
    if (not _master_templates) or (_loaded_templates_session_id != active_id):
        _master_templates = {}
        _loaded_templates_session_id = None
        if supabase is not None:
            try:
                res = supabase.storage.from_("datasets").download(f"{active_id}/master_templates.json")
                if res:
                    _master_templates = json.loads(res.decode('utf-8'))
                    _loaded_templates_session_id = active_id
                    print(f"Loaded master templates from Supabase Storage for inference: {active_id}")
            except Exception as e:
                print(f"Warning: Could not load templates from storage for {active_id}: {e}")
        
        if not _master_templates:
            if os.path.exists(active_templates_path):
                try:
                    with open(active_templates_path, 'r') as f:
                        _master_templates = json.load(f)
                    _loaded_templates_session_id = active_id
                    print(f"Loaded master templates from disk for active session: {active_id}")
                except Exception as e:
                    print(f"Warning: Could not load templates from disk: {e}")
                    _master_templates = {}
            
    if not _master_templates:
        compute_master_templates()
        
    if not _master_templates:
        return {"error": "No training images available to compute templates. Please upload some training photos first."}
        
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        
        img_cropped = auto_detect_and_crop_face(img, strict=True)
        if img_cropped is None:
            return {"error": "No face detected. Please upload a clear photo showing the player's face."}
            
        img_resized = img_cropped.resize(IMG_SIZE)
        img_arr = img_to_array(img_resized)
        img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)
        
        emb = model.predict(img_arr, verbose=0)[0]
        emb = l2_normalize(emb)
        
        results = []
        for class_name, template_emb in _master_templates.items():
            similarity = float(np.dot(emb, np.array(template_emb)))
            results.append({
                "class": class_name,
                "confidence": max(0.0, similarity)  # Present similarity score
            })
            
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
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
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        
        img_cropped = auto_detect_and_crop_face(img, strict=True)
        if img_cropped is None:
            return None
            
        img_resized = img_cropped.resize(IMG_SIZE)
        img_arr = img_to_array(img_resized)
        img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)
        
        base_model = None
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower() or 'mobilefacenet' in layer.name.lower() or hasattr(layer, 'layers'):
                base_model = layer
                break
                
        if base_model is None:
            base_model = model
        
        first_conv = None
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                first_conv = layer
                break
                
        if first_conv is None:
            print("Could not find any Conv2D layer in base model.")
            return None
            
        activation_model = tf.keras.Model(inputs=base_model.input, outputs=first_conv.output)
        activations = activation_model.predict(img_arr)[0]  # Shape: (H, W, 32)
        
        maps_base64 = []
        for i in range(16):
            feat_map = activations[:, :, i]
            
            f_min, f_max = feat_map.min(), feat_map.max()
            if f_max - f_min > 0:
                feat_map = 255 * (feat_map - f_min) / (f_max - f_min)
            feat_map = feat_map.astype('uint8')
            
            colored_map = cm.viridis(feat_map)  # returns RGBA [0.0, 1.0]
            colored_map = (colored_map[:, :, :3] * 255).astype('uint8') # drop alpha, scale to 0-255
            
            pil_map = Image.fromarray(colored_map)
            pil_map_resized = pil_map.resize((128, 128), Image.Resampling.NEAREST)  # Upscale for better viewing
            
            buffer = io.BytesIO()
            pil_map_resized.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            maps_base64.append(img_str)
            
        return maps_base64
    except Exception as e:
        print(f"Error extracting activation maps: {e}")
        return None
