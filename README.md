# Dynamic CNN Face Classifier

A lightweight, plug-and-play face recognition and classification engine. The system allows users to dynamically create isolated training sessions, add custom face classes, crawler-scrape images, evaluate models via real-time Stratified K-Fold validation, and analyze explainable CNN activation maps.

---

## 🏗️ Architecture & Hosting Flow

The system runs as a unified **FastAPI application** that exposes REST API endpoints and statically hosts the frontend single-page application (SPA).

```text
+-----------------------------------------------------------------+
|                         FASTAPI SERVER                          |
|                                                                 |
|   [ GET / ]         --> Serves Frontend statically (HTML/CSS/JS)|
|   [ POST /api/* ]   --> Exposes REST endpoints for the client   |
+--------------------------------+--------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|                     DYNAMIC PIPELINE ENGINE                     |
|                                                                 |
|   * Face Detector   --> OpenCV ResNet-10 Caffe SSD              |
|   * Feature Map     --> Frozen Keras MobileFaceNet (256-D)      |
|   * Classifier      --> Cosine Similarity (Dot Product Lookup)  |
+--------------------------------+--------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|                    SUPABASE CLOUD SERVICES                      |
|                                                                 |
|   * Database        --> Session profiles & K-Fold history metrics|
|   * Storage         --> Datasets, model weights, & master vectors|
+-----------------------------------------------------------------+
```

---

## 🧠 Deep Learning Pipeline

### Prediction / Inference Flow:
```text
   [ Input Image File / URL ]
                |
                v
     +---------------------+
     |  OpenCV ResNet-10   |  <-- Auto-downloads deploy.prototxt & caffe weights
     |   Caffe Face SSD    |
     +---------------------+
                |
                +----> Precise square face crop (padded to 130% & resized to 112x112)
                v
     +---------------------+
     |    MobileFaceNet    |  <-- Loaded from Supabase Storage / local cache;
     |  (Frozen Extractor) |      all layers set to non-trainable
     +---------------------+
                |
                +----> Generates L2-normalized 256-Dimensional embedding vector
                v
     +---------------------+
     |  Cosine Similarity  |  <-- Calculates dot product of embedding against
     |   Matcher Engine    |      each active Class Master Template
     +---------------------+
                |
                +----> Calculates: Score = Vector(Input) · Template(Class)
                v
   [ Ranked Probabilities & 16-Channel Viridis Activation Maps ]
```

### Key Mechanisms:
*   **Plug-and-Play Sessions**: Adding classes or retraining does not update neural network weights. Instead, it recalculates **Master Class Templates** (the average embedding vector of training photos).
*   **Memory Optimization**: Restricts TensorFlow thread pools (`intra/inter_op = 1`) to fit within 512MB RAM server constraints.
*   **Stratified K-Fold Validation**: Automatically splits active session training images into stratified splits, evaluates local templates on validation folds, and reports metrics.
*   **Explainable AI**: Renders visual representation of the first 16 filters of the first convolutional layer.

---

## 📂 Project Layout

```text
├── backend/
│   ├── main.py             # FastAPI router, CORS, session endpoints & static mounting
│   ├── model_utils.py      # Face cropping, MobileFaceNet embedding, similarity & training
│   ├── dataset_prep.py     # DuckDuckGo scraper & curated fallback url downloads
│   ├── requirements.txt    # Python dependencies (TensorFlow, OpenCV, Supabase, etc.)
│   └── sessions/           # Local folder mapping cached session assets
└── frontend/
    ├── index.html          # SPA dashboard markup
    ├── style.css           # Glassmorphism styling rules
    └── app.js              # State manager, Chart.js handler & fetch requests
```

---

## ⚙️ Setup & Execution

### 1. Environment
Configure a Supabase project containing a `sessions` table (columns: `id`, `display_name`, `classes`, `status`, `is_active`, `history`) and a public `datasets` storage bucket.

Create a `.env` file in the root of the project:
```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-service-role-or-anon-key
```

### 2. Run Application
Run the backend server. The frontend folder is automatically served statically at `http://127.0.0.1:8000/`.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

---

