import os
import json
import shutil
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from dataset_prep import prepare_dataset, DATASET_DIR, PLAYERS
from model_utils import train_cnn_model, run_inference, extract_activation_maps, STATUS_PATH, MODEL_PATH, CLASSES

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
    """Pre-loads the saved Keras model into memory at startup to speed up inference."""
    from model_utils import get_model
    print("Pre-loading model into memory cache...")
    get_model()

@app.get("/api/dataset-info")
def get_dataset_info():
    """Gets counts of downloaded images in the dataset folders."""
    info = {}
    for player in PLAYERS.keys():
        player_dir = os.path.join(DATASET_DIR, player)
        if os.path.exists(player_dir):
            files = [f for f in os.listdir(player_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            info[player] = len(files)
        else:
            info[player] = 0
    return {
        "status": "ready",
        "counts": info,
        "model_exists": os.path.exists(MODEL_PATH)
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

def run_train_task(epochs: int, batch_size: int):
    global task_state
    try:
        task_state["training"] = True
        train_cnn_model(epochs=epochs, batch_size=batch_size)
    except Exception as e:
        print(f"Error training model: {e}")
        # Log failure
        with open(STATUS_PATH, 'w') as f:
            json.dump({"status": "failed", "error": str(e)}, f)
    finally:
        task_state["training"] = False

@app.post("/api/train")
def trigger_train(background_tasks: BackgroundTasks, epochs: int = 10, batch_size: int = 4):
    """Triggers CNN model training in background."""
    global task_state
    if task_state["training"]:
        return {"status": "already_running", "message": "Model training is already running."}
        
    background_tasks.add_task(run_train_task, epochs, batch_size)
    return {"status": "started", "message": "Training started."}

@app.get("/api/status")
def get_status():
    """Polls overall download and training status."""
    status_data = {
        "downloading": task_state["downloading"],
        "download_summary": task_state["download_summary"],
        "training": task_state["training"],
        "model_exists": os.path.exists(MODEL_PATH),
        "keras_logs": None
    }
    
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, 'r') as f:
                status_data["keras_logs"] = json.load(f)
        except Exception:
            pass
            
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
    """Allows uploading custom images to build/enrich training datasets."""
    if player not in CLASSES:
        raise HTTPException(status_code=400, detail="Invalid footballer. Must be messi, yamal, or lewandowski.")
        
    player_dir = os.path.join(DATASET_DIR, player)
    os.makedirs(player_dir, exist_ok=True)
    
    # Save the file
    filename = f"upload_{int(os.path.getmtime(player_dir) * 1000)}_{file.filename}"
    filepath = os.path.join(player_dir, filename)
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check image validity using Pillow
        from PIL import Image
        img = Image.open(filepath)
        img.verify()
        
        # Convert to RGB to ensure clean JPG loading
        img = Image.open(filepath)
        img = img.convert('RGB')
        img.save(filepath, "JPEG")
        
        return {"status": "success", "message": f"Successfully uploaded image for {player.upper()}."}
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# Mount frontend files (fallback to index.html if route not matched)
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: Frontend static directory {frontend_dir} not found. Please create it.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
