import os
# Force Keras 2 legacy behavior for model loading compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import json
import shutil
import tempfile
import zipfile
import time
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional

from dataset_prep import prepare_dataset, DATASET_DIR, PLAYERS
from model_utils import (
    train_cnn_model, run_inference, extract_activation_maps, STATUS_PATH, MODEL_PATH,
    CLASSES, auto_detect_and_crop_face, get_active_classes, get_active_dataset_dir,
    get_active_templates_path, get_active_metadata_path, update_active_session_metadata,
    activate_session, list_sessions, get_session_details, SESSIONS_DIR, supabase, list_all_storage_files
)

app = FastAPI(title="Barca Footballer CNN Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

task_state = {
    "downloading": False,
    "download_summary": {},
    "training": False
}

@app.on_event("startup")
def startup_event():
    """Starts heavy initialization in a background thread so the port binds immediately."""
    import threading

    def _init_background():
        from model_utils import get_model, MODEL_PATH, _load_initial_templates
        import os

        print("Loading master templates...")
        try:
            _load_initial_templates()
        except Exception as e:
            print(f"Warning: Could not load initial templates: {e}")

        print("Pre-loading model into memory cache...")
        model = get_model()
        if model is not None and not os.path.exists(MODEL_PATH):
            try:
                model.save(MODEL_PATH)
                print(f"Saved initial model skeleton to {MODEL_PATH}")
            except Exception as e:
                print(f"Warning: Could not save model skeleton: {e}")

        print("Background initialization complete.")

    thread = threading.Thread(target=_init_background, daemon=True)
    thread.start()
    print("Server ready. Background initialization started in separate thread.")


@app.get("/api/dataset-info")
def get_dataset_info():
    active_classes = get_active_classes()
    from model_utils import get_active_session_details
    active_id, _, _ = get_active_session_details()
    session_details = get_session_details(active_id)
    
    info = {}
    for player in active_classes:
        if supabase is not None:
            try:
                files = list_all_storage_files("datasets", f"{active_id}/{player}")
                count = len([f for f in files if f["name"].lower().endswith(('.jpg', '.jpeg', '.png'))]) if files else 0
                info[player] = count
            except Exception:
                info[player] = 0
        else:
            info[player] = 0
            
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
        
        inference_res = run_inference(contents)
        if "error" in inference_res:
            return JSONResponse(status_code=400, content=inference_res)
            
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
    active_classes = get_active_classes()
    from model_utils import get_active_session_details, compute_master_templates
    active_id, _, _ = get_active_session_details()
    
    if player not in active_classes:
        raise HTTPException(status_code=400, detail=f"Invalid footballer class '{player}'. Must be one of: {active_classes}")
        
    filename = f"upload_{int(time.time() * 1000)}.jpg"
    try:
        contents = await file.read()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_cropped = auto_detect_and_crop_face(img, strict=False)
        
        buffer = io.BytesIO()
        img_cropped.save(buffer, format="JPEG")
        
        if supabase is not None:
            supabase.storage.from_("datasets").upload(
                path=f"{active_id}/{player}/{filename}",
                file=buffer.getvalue(),
                file_options={"content-type": "image/jpeg"}
            )
            
            # Recalculate templates if already trained
            details = get_session_details(active_id)
            if details and details.get("status") == "completed":
                try:
                    compute_master_templates()
                except Exception:
                    pass
                    
        return {"status": "success", "message": f"Successfully uploaded and cropped image for {player.upper()}."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

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
    session_id = session_data.id
    if supabase is not None:
        try:
            res = supabase.table("sessions").select("id").eq("id", session_id).execute()
            if res.data:
                raise HTTPException(status_code=400, detail=f"Session with ID '{session_id}' already exists.")
        except Exception as e:
            if "already exists" in str(e):
                raise HTTPException(status_code=400, detail=f"Session with ID '{session_id}' already exists.")
                
    metadata = {
        "id": session_id,
        "display_name": session_data.display_name,
        "classes": session_data.classes,
        "status": "untrained",
        "is_active": False,
        "history": {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
    }
    
    if supabase is not None:
        try:
            supabase.table("sessions").insert(metadata).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create session in Supabase: {str(e)}")
            
    return metadata

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    from model_utils import get_active_session_details
    active_id, _, _ = get_active_session_details()
    if active_id == session_id:
        raise HTTPException(status_code=400, detail="Cannot delete the currently active session. Activate another session first.")
        
    details = get_session_details(session_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    try:
        if supabase is not None:
            supabase.table("sessions").delete().eq("id", session_id).execute()
            
            for class_name in details.get("classes", []):
                storage_path = f"{session_id}/{class_name}"
                try:
                    files = list_all_storage_files("datasets", storage_path)
                    if files:
                        to_remove = [f"{storage_path}/{f['name']}" for f in files]
                        # Supabase remove takes chunks of max 100, but let's send them
                        # in chunks to prevent RLS/parameter overflows
                        for chunk_idx in range(0, len(to_remove), 100):
                            supabase.storage.from_("datasets").remove(to_remove[chunk_idx:chunk_idx+100])
                except Exception as e:
                    print(f"Error removing files during session delete: {e}")
            try:
                supabase.storage.from_("datasets").remove([f"{session_id}/master_templates.json"])
            except Exception:
                pass
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
    display_message: Optional[str] = None

@app.post("/api/sessions/{session_id}/rename")
def rename_session(session_id: str, data: SessionRenameSchema):
    details = get_session_details(session_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    try:
        if supabase is not None:
            history = details.get("history", {})
            if not isinstance(history, dict):
                history = {}
            if data.display_message is not None:
                history["display_message"] = data.display_message
            
            res = supabase.table("sessions").update({
                "display_name": data.display_name,
                "history": history
            }).eq("id", session_id).execute()
            
            if res.data:
                return {"status": "success", "message": "Successfully updated session config.", "metadata": res.data[0]}
        return {"status": "success", "message": "Successfully renamed session.", "metadata": details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")

class SessionClassesSchema(BaseModel):
    classes: List[str]

@app.post("/api/sessions/{session_id}/classes")
def update_session_classes(session_id: str, data: SessionClassesSchema):
    details = get_session_details(session_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    try:
        old_classes = details.get("classes", [])
        new_classes = data.classes

        if supabase is not None:
            # Rename storage folders when class names change
            for old_name, new_name in zip(old_classes, new_classes):
                if old_name != new_name:
                    try:
                        old_prefix = f"{session_id}/{old_name}"
                        new_prefix = f"{session_id}/{new_name}"
                        from model_utils import list_all_storage_files
                        files = list_all_storage_files("datasets", old_prefix)
                        for f_info in files:
                            fname = f_info.get("name")
                            if not fname:
                                continue
                            old_path = f"{old_prefix}/{fname}"
                            new_path = f"{new_prefix}/{fname}"
                            try:
                                file_bytes = supabase.storage.from_("datasets").download(old_path)
                                if file_bytes:
                                    mime = "image/png" if fname.lower().endswith(".png") else "image/jpeg"
                                    supabase.storage.from_("datasets").upload(
                                        path=new_path,
                                        file=file_bytes,
                                        file_options={"content-type": mime, "x-upsert": "true", "upsert": "true"}
                                    )
                                    supabase.storage.from_("datasets").remove([old_path])
                            except Exception:
                                pass
                        print(f"Renamed class storage folder: '{old_name}' -> '{new_name}' in session '{session_id}'")
                    except Exception as re:
                        print(f"Warning: Could not rename storage folder for class '{old_name}': {re}")

            res = supabase.table("sessions").update({"classes": new_classes}).eq("id", session_id).execute()
            if res.data:
                import model_utils
                model_utils._active_session_id = None
                return {"status": "success", "message": "Successfully updated session classes config.", "metadata": res.data[0]}
        return {"status": "success", "message": "Successfully updated session classes config.", "metadata": details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session classes: {str(e)}")

@app.post("/api/sessions/{session_id}/upload")
async def upload_session_image(session_id: str, player: str = Form(...), file: UploadFile = File(...)):
    session_details = get_session_details(session_id)
    if not session_details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    classes = session_details.get("classes", [])
    if player not in classes:
        raise HTTPException(status_code=400, detail=f"Invalid class '{player}'. Must be one of: {classes}")
    try:
        contents = await file.read()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_cropped = auto_detect_and_crop_face(img, strict=False)
        
        buffer = io.BytesIO()
        img_cropped.save(buffer, format="JPEG")
        
        filename = f"upload_{int(time.time() * 1000)}.jpg"
        if supabase is not None:
            supabase.storage.from_("datasets").upload(
                path=f"{session_id}/{player}/{filename}",
                file=buffer.getvalue(),
                file_options={"content-type": "image/jpeg"}
            )
            
            # Recalculate templates if already trained
            if session_details.get("status") == "completed":
                try:
                    from model_utils import compute_master_templates
                    compute_master_templates()
                except Exception:
                    pass
        return {"status": "success", "message": f"Successfully uploaded and cropped image for {player.upper()} in session {session_id}."}
    except Exception as e:
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
def export_session(session_id: str, background_tasks: BackgroundTasks):
    details = get_session_details(session_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    temp_dir = tempfile.TemporaryDirectory()
    local_path = temp_dir.name
    
    try:
        os.makedirs(os.path.join(local_path, "dataset"), exist_ok=True)
        with open(os.path.join(local_path, "metadata.json"), 'w') as f:
            json.dump(details, f)
            
        for class_name in details.get("classes", []):
            os.makedirs(os.path.join(local_path, "dataset", class_name), exist_ok=True)
            if supabase is not None:
                storage_path = f"{session_id}/{class_name}"
                try:
                    files = list_all_storage_files("datasets", storage_path)
                    if files:
                        for f_info in files:
                            fname = f_info["name"]
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                                fdata = supabase.storage.from_("datasets").download(f"{storage_path}/{fname}")
                                if fdata:
                                    with open(os.path.join(local_path, "dataset", class_name, fname), 'wb') as img_f:
                                        img_f.write(fdata)
                except Exception as e:
                    print(f"Error exporting files: {e}")
                    
        if supabase is not None:
            try:
                templates_data = supabase.storage.from_("datasets").download(f"{session_id}/master_templates.json")
                if templates_data:
                    with open(os.path.join(local_path, "master_templates.json"), 'wb') as mt_f:
                        mt_f.write(templates_data)
            except Exception:
                pass
                
        zip_dest = os.path.join(tempfile.gettempdir(), f"{session_id}_export_{int(time.time())}")
        archive_path = shutil.make_archive(zip_dest, 'zip', local_path)
        
        background_tasks.add_task(os.remove, archive_path)
        
        return FileResponse(
            path=archive_path,
            filename=f"{session_id}_session.zip",
            media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export session: {str(e)}")
    finally:
        try:
            temp_dir.cleanup()
        except Exception:
            pass

@app.post("/api/sessions/import")
async def import_session(file: UploadFile = File(...)):
    filename = file.filename
    if not filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be a .zip file.")
        
    session_id = filename.replace('_session.zip', '').replace('.zip', '').lower()
    import re
    session_id = re.sub(r'[^a-z0-9_-]', '_', session_id)
    
    if supabase is not None:
        try:
            res = supabase.table("sessions").select("id").eq("id", session_id).execute()
            if res.data:
                session_id = f"{session_id}_{int(time.time())}"
        except Exception:
            pass
            
    temp_dir = tempfile.TemporaryDirectory()
    local_path = temp_dir.name
    temp_zip = os.path.join(local_path, "import.zip")
    
    try:
        with open(temp_zip, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(local_path)
            
        metadata_path = os.path.join(local_path, "metadata.json")
        if not os.path.exists(metadata_path):
            for root, dirs, files in os.walk(local_path):
                if "metadata.json" in files:
                    metadata_path = os.path.join(root, "metadata.json")
                    break
                    
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=400, detail="Uploaded zip does not contain a valid session metadata.json.")
            
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            
        meta["id"] = session_id
        meta["is_active"] = False
        
        if supabase is not None:
            supabase.table("sessions").insert(meta).execute()
            
            for class_name in meta.get("classes", []):
                class_folder = os.path.join(os.path.dirname(metadata_path), "dataset", class_name)
                if os.path.exists(class_folder):
                    for img_name in os.listdir(class_folder):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(class_folder, img_name)
                            try:
                                with open(img_path, 'rb') as img_data:
                                    from model_utils import sanitize_filename
                                    clean_img_name = sanitize_filename(img_name)
                                    supabase.storage.from_("datasets").upload(
                                        path=f"{session_id}/{class_name}/{clean_img_name}",
                                        file=img_data.read(),
                                        file_options={"content-type": "image/jpeg" if img_name.lower().endswith(('.jpg', '.jpeg')) else "image/png"}
                                    )
                            except Exception:
                                pass
                                
            templates_path = os.path.join(os.path.dirname(metadata_path), "master_templates.json")
            if os.path.exists(templates_path):
                try:
                    with open(templates_path, 'rb') as mt_data:
                        supabase.storage.from_("datasets").upload(
                            path=f"{session_id}/master_templates.json",
                            file=mt_data.read(),
                            file_options={"content-type": "application/json", "x-upsert": "true", "upsert": "true"}
                        )
                except Exception:
                    pass
                    
        return {"status": "success", "message": f"Successfully imported session '{session_id}'.", "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import session: {str(e)}")
    finally:
        try:
            temp_dir.cleanup()
        except Exception:
            pass

@app.get("/api/sessions/{session_id}/dataset/{class_name}")
def list_headshots(session_id: str, class_name: str):
    session_details = get_session_details(session_id)
    if not session_details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    classes = session_details.get("classes", [])
    if class_name not in classes:
        raise HTTPException(status_code=400, detail=f"Class '{class_name}' is not part of session '{session_id}'.")
        
    if supabase is not None:
        try:
            storage_path = f"{session_id}/{class_name}"
            files = list_all_storage_files("datasets", storage_path)
            if files:
                return sorted([f["name"] for f in files if f["name"].lower().endswith(('.jpg', '.jpeg', '.png'))])
        except Exception as e:
            print(f"Error listing headshots from storage: {e}")
    return []

@app.get("/api/sessions/{session_id}/dataset/{class_name}/{filename}")
def serve_headshot(session_id: str, class_name: str, filename: str):
    filename = os.path.basename(filename)
    if supabase is not None:
        try:
            storage_path = f"{session_id}/{class_name}/{filename}"
            file_bytes = supabase.storage.from_("datasets").download(storage_path)
            if file_bytes:
                media_type = "image/png" if filename.lower().endswith('.png') else "image/jpeg"
                return Response(content=file_bytes, media_type=media_type)
        except Exception as e:
            print(f"Error serving headshot from storage: {e}")
    raise HTTPException(status_code=404, detail="Headshot file not found.")

class BulkDeleteSchema(BaseModel):
    filenames: List[str]

@app.delete("/api/sessions/{session_id}/dataset/{class_name}")
def delete_headshots(session_id: str, class_name: str, data: BulkDeleteSchema):
    session_details = get_session_details(session_id)
    if not session_details:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        
    classes = session_details.get("classes", [])
    if class_name not in classes:
        raise HTTPException(status_code=400, detail=f"Class '{class_name}' is not part of session '{session_id}'.")
        
    deleted_count = 0
    if supabase is not None:
        to_remove = [f"{session_id}/{class_name}/{os.path.basename(f)}" for f in data.filenames]
        try:
            res = supabase.storage.from_("datasets").remove(to_remove)
            if res:
                deleted_count = len(res)
        except Exception as e:
            print(f"Error removing files from Supabase Storage: {e}")
            
    if deleted_count > 0:
        from model_utils import get_active_session_details, compute_master_templates
        active_id, _, _ = get_active_session_details()
        if active_id == session_id:
            try:
                compute_master_templates()
            except Exception as e:
                print(f"Warning: Could not re-compute templates after deletion: {e}")
                
    return {"status": "success", "message": f"Successfully deleted {deleted_count} headshots from {class_name}.", "deleted_count": deleted_count}

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: Frontend static directory {frontend_dir} not found. Please create it.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
