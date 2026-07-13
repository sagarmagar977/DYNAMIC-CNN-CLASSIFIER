import os
import json
import shutil
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
import time
import zipfile

from dataset_prep import prepare_dataset, DATASET_DIR, PLAYERS
from model_utils import (
    train_cnn_model, run_inference, extract_activation_maps, STATUS_PATH, MODEL_PATH,
    CLASSES, auto_detect_and_crop_face, get_active_classes, get_active_dataset_dir,
    get_active_templates_path, get_active_metadata_path, update_active_session_metadata,
    activate_session, list_sessions, get_session_details, SESSIONS_DIR
)

app = FastAPI(title="Barca Footballer CNN Classifier")

# CORS middleware for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global task state tracking
task_state = {
    "downloading": False,
    "download_summary": {},
    "training": False
}

@app.on_event("startup")
def startup_event():
    """Performs legacy migrations, pre-loads the model, and initializes the active session."""
    from model_utils import get_model, compute_master_templates, MODEL_PATH, auto_migrate_legacy_dataset
    import os
    
    print("Running auto-migration of legacy dataset...")
    try:
        auto_migrate_legacy_dataset()
    except Exception as e:
        print(f"Warning: Could not complete legacy dataset migration: {e}")
        
    print("Pre-loading model into memory cache...")
    model = get_model()
    if model is not None and not os.path.exists(MODEL_PATH):
        try:
            model.save(MODEL_PATH)
            print(f"Saved initial model skeleton to {MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not save model skeleton: {e}")
            
    print("Pre-computing master templates for active session...")
    try:
        compute_master_templates()
    except Exception as e:
        print(f"Warning: Could not pre-compute master templates on startup: {e}")

@app.get("/api/dataset-info")
def get_dataset_info():
    """Gets counts of images in the dataset folders for the active session."""
    active_classes = get_active_classes()
    active_dataset_dir = get_active_dataset_dir()
    
    info = {}
    for player in active_classes:
        player_dir = os.path.join(active_dataset_dir, player)
        if os.path.exists(player_dir):
            files = [f for f in os.listdir(player_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            info[player] = len(files)
        else:
            info[player] = 0
            
    # Get active session ID
    from model_utils import get_active_session_details
    active_id, _, _ = get_active_session_details()
    session_details = get_session_details(active_id)
    
    return {
        "status": "ready",
        "active_session_id": active_id,
        "display_name": session_details.get("display_name", active_id) if session_details else active_id,
        "counts": info,
        "model_exists": os.path.exists(MODEL_PATH),
        "session_status": session_details.get("status", "untrained") if session_details else "untrained"
    }

def run_download_task():
    global task_state
    try:
        task_state["downloading"] = True
        summary = prepare_dataset(max_images_per_player=15)
        task_state["download_summary"] = summary
    except Exception as e:
        print(f"Error downloading dataset: {e}")
    finally:
        task_state["downloading"] = False

@app.post("/api/download-dataset")
def trigger_download(background_tasks: BackgroundTasks):
    """Triggers download of footballer images in background."""
    global task_state
    if task_state["downloading"]:
        return {"status": "already_running", "message": "Download is already running in the background."}
    
    background_tasks.add_task(run_download_task)
    return {"status": "started", "message": "Dataset download started in background."}

def run_train_task(folds: int):
    global task_state
    try:
        task_state["training"] = True
        train_cnn_model(epochs=folds)
    except Exception as e:
        print(f"Error training model: {e}")
        # Log failure
        with open(STATUS_PATH, 'w') as f:
            json.dump({"status": "failed", "error": str(e)}, f)
    finally:
        task_state["training"] = False

@app.post("/api/train")
def trigger_train(background_tasks: BackgroundTasks, folds: int = 5):
    """Triggers dynamic K-Fold validation in background for the active session."""
    from model_utils import get_active_session_details
    active_id, _, _ = get_active_session_details()
    return trigger_session_train(active_id, background_tasks, folds)

@app.get("/api/status")
def get_status():
    """Polls overall download and training status, reading active session logs if available."""
    status_data = {
        "downloading": task_state["downloading"],
        "download_summary": task_state["download_summary"],
        "training": task_state["training"],
        "model_exists": os.path.exists(MODEL_PATH),
        "keras_logs": None
    }
    
    # Read active session status from metadata.json
    from model_utils import get_active_session_details
    try:
        active_id, _, _ = get_active_session_details()
        details = get_session_details(active_id)
        if details:
            status_data["keras_logs"] = {
                "status": details.get("status", "untrained"),
                "current_epoch": len(details.get("history", {}).get("loss", [])) if details.get("status") == "completed" else 0,
                "total_epochs": len(details.get("history", {}).get("loss", [])) if details.get("status") == "completed" else 5,
                "history": details.get("history", {})
            }
            # If currently training, fall back to STATUS_PATH for live epoch-by-epoch updates
            if task_state["training"] and os.path.exists(STATUS_PATH):
                with open(STATUS_PATH, 'r') as f:
                    status_data["keras_logs"] = json.load(f)
    except Exception as e:
        print(f"Warning: Could not compile live status: {e}")
        
    return status_data

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """Receives a file, runs prediction, and returns probabilities and activation maps."""
    if not os.path.exists(MODEL_PATH):
        return JSONResponse(
            status_code=400,
            content={"error": "Model not trained yet. Please train the CNN model first."}
        )
        
    try:
        contents = await file.read()
        
        # Run classification model
        inference_res = run_inference(contents)
        if "error" in inference_res:
            return JSONResponse(status_code=400, content=inference_res)
            
        # Run activation map extractor
        activation_maps = extract_activation_maps(contents)
        
        return {
            "predictions": inference_res["predictions"],
            "cropped_face": inference_res.get("cropped_face"),
            "activation_maps": activation_maps
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal prediction error: {str(e)}"})

@app.post("/api/upload-training-image")
async def upload_training_image(player: str = Form(...), file: UploadFile = File(...)):
    """Allows uploading custom images to active training datasets, cropping them immediately."""
    active_classes = get_active_classes()
    active_dataset_dir = get_active_dataset_dir()
    
    if player not in active_classes:
        raise HTTPException(status_code=400, detail=f"Invalid footballer class '{player}'. Must be one of: {active_classes}")
        
    player_dir = os.path.join(active_dataset_dir, player)
    os.makedirs(player_dir, exist_ok=True)
    
    # Save the file
    filename = f"upload_{int(time.time() * 1000)}_{file.filename}"
    filepath = os.path.join(player_dir, filename)
    
    try:
        # Read file to memory
        contents = await file.read()
        
        # Load and convert image
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents))
        img = img.convert('RGB')
        
        # Run face cropping logic immediately on upload
        img_cropped = auto_detect_and_crop_face(img, strict=False)
        
        # Save the cropped image
        img_cropped.save(filepath, "JPEG")
        
        return {"status": "success", "message": f"Successfully uploaded and cropped image for {player.upper()}."}
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# ==================== DYNAMIC MULTI-SESSION ENDPOINTS ====================

@app.get("/api/sessions")
def get_sessions_endpoint():
    """Lists all available model manager sessions/profiles."""
    return list_sessions()

class SessionCreateSchema(BaseModel):
    id: str
    display_name: str
    classes: List[str]

    @validator('id')
    def validate_id(cls, v):
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('ID must be alphanumeric, containing only letters, numbers, hyphens, and underscores.')
        return v.lower()

@app.post("/api/sessions")
def create_session(session_data: SessionCreateSchema):
    """Creates a new session directory and initializes its metadata."""
    session_id = session_data.id
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if os.path.exists(session_dir):
        raise HTTPException(status_code=400, detail=f"Session with ID '{session_id}' already exists.")
        
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(os.path.join(session_dir, "dataset"), exist_ok=True)
    
    # Initialize metadata.json
    metadata = {
        "id": session_id,
        "display_name": session_data.display_name,
        "classes": session_data.classes,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "untrained",
        "is_active": False,
        "history": {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
    }
    
    # Create subdirectories for each class inside the dataset folder
    for class_name in session_data.classes:
        os.makedirs(os.path.join(session_dir, "dataset", class_name), exist_ok=True)
        
    with open(os.path.join(session_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f)
        
    return metadata

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Deletes a session directory and all its files."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    # Check if the active session is being deleted
    from model_utils import get_active_session_details
    active_id, _, _ = get_active_session_details()
    if active_id == session_id:
        raise HTTPException(status_code=400, detail="Cannot delete the currently active session. Activate another session first.")
        
    try:
        shutil.rmtree(session_dir)
        return {"status": "success", "message": f"Successfully deleted session '{session_id}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@app.post("/api/sessions/{session_id}/activate")
def activate_session_endpoint(session_id: str):
    """Switches the active session profile."""
    try:
        activate_session(session_id)
        details = get_session_details(session_id)
        return {"status": "success", "message": f"Session '{session_id}' activated successfully.", "metadata": details}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate session: {str(e)}")

class SessionRenameSchema(BaseModel):
    display_name: str

@app.post("/api/sessions/{session_id}/rename")
def rename_session(session_id: str, data: SessionRenameSchema):
    """Renames the display name of a session profile."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    metadata_path = os.path.join(session_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    try:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        meta["display_name"] = data.display_name
        with open(metadata_path, 'w') as f:
            json.dump(meta, f)
        return {"status": "success", "message": f"Successfully renamed session display name to '{data.display_name}'.", "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename session: {str(e)}")

class SessionClassesSchema(BaseModel):
    classes: List[str]

@app.post("/api/sessions/{session_id}/classes")
def update_session_classes(session_id: str, data: SessionClassesSchema):
    """Updates the class list configuration for a session."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    metadata_path = os.path.join(session_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    try:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            
        old_classes = meta.get("classes", [])
        new_classes = data.classes
        
        # Update metadata classes
        meta["classes"] = new_classes
        
        # Ensure directories exist for all new classes
        for class_name in new_classes:
            os.makedirs(os.path.join(session_dir, "dataset", class_name), exist_ok=True)
            
        # Try to rename folders if a rename happened to keep files
        # We can map by index (e.g. index 0 in old matches index 0 in new)
        for idx, new_name in enumerate(new_classes):
            if idx < len(old_classes):
                old_name = old_classes[idx]
                if old_name != new_name:
                    old_path = os.path.join(session_dir, "dataset", old_name)
                    new_path = os.path.join(session_dir, "dataset", new_name)
                    if os.path.exists(old_path) and not os.path.exists(new_path):
                        try:
                            os.rename(old_path, new_path)
                            print(f"Renamed class folder from {old_name} to {new_name}")
                        except Exception as e:
                            print(f"Failed to rename class folder: {e}")
                            
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(meta, f)
            
        # Force reload in memory active details if needed
        import model_utils
        model_utils._active_session_id = None
        
        return {"status": "success", "message": "Successfully updated session classes config.", "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session classes: {str(e)}")

@app.post("/api/sessions/{session_id}/upload")
async def upload_session_image(session_id: str, player: str = Form(...), file: UploadFile = File(...)):
    """Uploads a training image to a specific session's class dataset directory, auto-cropping it."""
    session_details = get_session_details(session_id)
    if not session_details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    classes = session_details.get("classes", [])
    if player not in classes:
        raise HTTPException(status_code=400, detail=f"Invalid class '{player}'. Must be one of: {classes}")
        
    player_dir = os.path.join(SESSIONS_DIR, session_id, "dataset", player)
    os.makedirs(player_dir, exist_ok=True)
    
    filename = f"upload_{int(time.time() * 1000)}_{file.filename}"
    filepath = os.path.join(player_dir, filename)
    
    try:
        contents = await file.read()
        
        # Load and convert image
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents))
        img = img.convert('RGB')
        
        # Run face cropping logic
        img_cropped = auto_detect_and_crop_face(img, strict=False)
        img_cropped.save(filepath, "JPEG")
        
        return {"status": "success", "message": f"Successfully uploaded and cropped image for {player.upper()} in session {session_id}."}
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def run_session_train_task(session_id: str, folds: int):
    global task_state
    try:
        task_state["training"] = True
        activate_session(session_id)
        train_cnn_model(epochs=folds)
    except Exception as e:
        print(f"Error training session {session_id}: {e}")
    finally:
        task_state["training"] = False

@app.post("/api/sessions/{session_id}/train")
def trigger_session_train(session_id: str, background_tasks: BackgroundTasks, folds: int = 5):
    """Triggers K-Fold validation training for a specific session."""
    global task_state
    details = get_session_details(session_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    if task_state["training"]:
        return {"status": "already_running", "message": "Model training/validation is already running."}
        
    background_tasks.add_task(run_session_train_task, session_id, folds)
    return {"status": "started", "message": f"K-Fold validation started for session '{session_id}'."}

@app.get("/api/sessions/{session_id}/export")
def export_session(session_id: str):
    """Exports a session directory as a zip file download."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    # Create zip in a temporary location inside workspace
    temp_dir = os.path.abspath(os.path.join(SESSIONS_DIR, "..", "temp_exports"))
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, f"{session_id}_export")
    
    try:
        archive_path = shutil.make_archive(zip_path, 'zip', session_dir)
        return FileResponse(
            path=archive_path,
            filename=f"{session_id}_session.zip",
            media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export session: {str(e)}")

@app.post("/api/sessions/import")
async def import_session(file: UploadFile = File(...)):
    """Imports a session profile from an uploaded zip file."""
    filename = file.filename
    if not filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be a .zip file.")
        
    session_id = filename.replace('_session.zip', '').replace('.zip', '').lower()
    import re
    session_id = re.sub(r'[^a-z0-9_-]', '_', session_id)
    
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if os.path.exists(session_dir):
        session_id = f"{session_id}_{int(time.time())}"
        session_dir = os.path.join(SESSIONS_DIR, session_id)
        
    os.makedirs(session_dir, exist_ok=True)
    
    temp_zip_path = os.path.join(SESSIONS_DIR, f"temp_import_{session_id}.zip")
    try:
        with open(temp_zip_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(session_dir)
            
        metadata_path = os.path.join(session_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            found_meta = False
            for root, dirs, files in os.walk(session_dir):
                if "metadata.json" in files:
                    sub_meta = os.path.join(root, "metadata.json")
                    shutil.copy2(sub_meta, metadata_path)
                    sub_dataset = os.path.join(root, "dataset")
                    if os.path.exists(sub_dataset):
                        shutil.copytree(sub_dataset, os.path.join(session_dir, "dataset"), dirs_exist_ok=True)
                    sub_templates = os.path.join(root, "master_templates.json")
                    if os.path.exists(sub_templates):
                        shutil.copy2(sub_templates, os.path.join(session_dir, "master_templates.json"))
                    found_meta = True
                    break
            if not found_meta:
                shutil.rmtree(session_dir)
                raise HTTPException(status_code=400, detail="Uploaded zip does not contain a valid session metadata.json.")
                
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            
        meta["id"] = session_id
        meta["is_active"] = False
        
        with open(metadata_path, 'w') as f:
            json.dump(meta, f)
            
        return {"status": "success", "message": f"Successfully imported session '{session_id}'.", "metadata": meta}
    except Exception as e:
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=f"Failed to import session: {str(e)}")
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

@app.get("/api/sessions/{session_id}/dataset/{class_name}")
def list_headshots(session_id: str, class_name: str):
    """Lists all cropped headshot filenames for a given session and class."""
    session_details = get_session_details(session_id)
    if not session_details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    classes = session_details.get("classes", [])
    if class_name not in classes:
        raise HTTPException(status_code=400, detail=f"Class '{class_name}' is not part of session '{session_id}'.")
        
    class_dir = os.path.join(SESSIONS_DIR, session_id, "dataset", class_name)
    if not os.path.exists(class_dir):
        return []
        
    files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return sorted(files)

@app.get("/api/sessions/{session_id}/dataset/{class_name}/{filename}")
def serve_headshot(session_id: str, class_name: str, filename: str):
    """Serves a specific cropped headshot image file."""
    filename = os.path.basename(filename)
    filepath = os.path.join(SESSIONS_DIR, session_id, "dataset", class_name, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Headshot file not found.")
    return FileResponse(filepath)

class BulkDeleteSchema(BaseModel):
    filenames: List[str]

@app.delete("/api/sessions/{session_id}/dataset/{class_name}")
def delete_headshots(session_id: str, class_name: str, data: BulkDeleteSchema):
    """Deletes list of specified headshot filenames from a session's class folder."""
    session_details = get_session_details(session_id)
    if not session_details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    classes = session_details.get("classes", [])
    if class_name not in classes:
        raise HTTPException(status_code=400, detail=f"Class '{class_name}' is not part of session '{session_id}'.")
        
    class_dir = os.path.join(SESSIONS_DIR, session_id, "dataset", class_name)
    if not os.path.exists(class_dir):
        return {"status": "success", "message": "Class directory is already empty.", "deleted_count": 0}
        
    deleted_count = 0
    for filename in data.filenames:
        clean_filename = os.path.basename(filename)
        filepath = os.path.join(class_dir, clean_filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting file {filepath}: {e}")
                
    templates_path = os.path.join(SESSIONS_DIR, session_id, "master_templates.json")
    if os.path.exists(templates_path) and deleted_count > 0:
        from model_utils import get_active_session_details, compute_master_templates
        active_id, _, _ = get_active_session_details()
        if active_id == session_id:
            try:
                compute_master_templates()
            except Exception as e:
                print(f"Warning: Could not re-compute templates after deletion: {e}")
                
    return {"status": "success", "message": f"Successfully deleted {deleted_count} headshots from {class_name}.", "deleted_count": deleted_count}

# Mount frontend files (fallback to index.html if route not matched)
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: Frontend static directory {frontend_dir} not found. Please create it.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
