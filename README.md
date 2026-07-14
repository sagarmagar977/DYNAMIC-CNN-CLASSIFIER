# Dynamic CNN Face Classifier

A lightweight, plug-and-play face recognition engine that implements the same mathematical paradigm used in modern biometric systems like **Apple Face ID**:

*   **One-Shot Enrollment**: Traditional image classifiers require retraining the neural network's final layers to recognize new faces. This system, like Face ID, bypasses retraining by mapping facial features to a **256-D face embedding vector** via a frozen, pre-trained feature extractor (MobileFaceNet).
*   **Vector Comparisons**: During setup, the average embedding vector of a user's face is saved as a **Master Template**. During prediction, the scanned face is mapped to an embedding and compared using **Cosine Similarity** (calculating the dot product of the vectors). If the score exceeds a threshold, the face is recognized.
*   **Cloud Architecture & Explainability**: The entire pipeline is wrapped in a unified **FastAPI server** that statically hosts frontend dashboard files, communicates with **Supabase** for database/storage cloud synchronization, and renders real-time Stratified K-Fold validation metrics and explainable CNN activation maps.
*   **Real-World vs. Local Differences**: While consumer hardware uses active infrared 3D depth maps and local hardware-isolated Secure Enclave storage for anti-spoofing and security, this project demonstrates the identical algorithmic flow using standard 2D images and cloud databases.

---

## 🧠 Deep Learning Pipeline

### 1. Pre-Processing: OpenCV DNN Face Cropping (Applied to both Train & Test)
Before any model calculations occur, both the training dataset and test/inference images are processed through the same face alignment pipeline:
*   **Face Detection**: The raw image is passed to an **OpenCV ResNet-10 Caffe SSD face detector** (automatically downloads config and model weights).
*   **Precision Cropping**: The detector crops the face area with a center-seeking square, padded to 130% to preserve crucial structural jaw/hair bounds, and resizes it to 112×112px.
*   **Unified Inputs**: This ensures that both the generated master templates (during training) and the query vectors (during testing) are computed strictly from normalized, cropped facial areas, eliminating background noise.

### 2. How "Training" Works (K-Fold Validation & Template Generation)
Instead of standard training via backpropagation, the neural network backbone (MobileFaceNet) is completely frozen to prevent overfitting and fit constrained server environments:
1.  **Stratified K-Fold Cross-Validation**: Splits the pre-processed dataset embeddings into stratified folds to test performance.
2.  **Master Template Math**: For each class, the system calculates the **Master Vector Template** by computing the mean of all training embeddings and then L2-normalizing it.
3.  **Fold Evaluation**: The validation split's embeddings are matched against these temporary templates to calculate validation loss ($1 - \text{similarity}$) and accuracy in real-time.
4.  **Final Generation**: Once cross-validation completes, the system generates and saves the final Master Templates to Supabase.

### 3. How "Prediction" Works (On-the-Fly Cosine Similarity)
The prediction pipeline matches the query face embedding against the stored class master templates:

```text
   [ Input Image File / URL ]
                |
                v
     +---------------------+
     |  OpenCV ResNet-10   |  <-- Face detection utilizing a Caffe SSD model;
     |   Caffe Face SSD    |      auto-downloads deploy.prototxt and weights.
     +---------------------+
                |
                +----> Precise square face crop (padded to 130% & resized to 112x112)
                v
     +---------------------+
     |    MobileFaceNet    |  <-- Cached feature extractor model;
     |  (Frozen Extractor) |      all layers set to non-trainable.
     +---------------------+
                |
                +----> Generates L2-normalized 256-Dimensional embedding vector.
                v
     +---------------------+
     |  Cosine Similarity  |  <-- Calculates dot product of embedding against
     |   Matcher Engine    |      each active Class Master Template.
     +---------------------+
                |
                +----> Calculates: Score = Vector(Input) · Template(Class)
                v
   [ Ranked Cosine Similarities (Confidence Scores) & 16-Channel XAI Maps ]
```

### Core Deep Learning Mechanisms:
*   **Plug-and-Play Sessions**: Adding classes does not update neural network weights. Instead, it recalculates **Master Class Templates** (the mean embedding vector of all training embeddings).
*   **Explainability (XAI)**: Renders the first 16 convolutional channel activation maps using Matplotlib's viridis color scheme to visualize structural features (edges, textures).

---

## 🏗️ Hosting & Architecture

The application runs as a unified **FastAPI server** that exposes REST API endpoints and statically hosts the frontend single-page application (SPA).

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

## 📂 Project Layout

```text
├── backend/
│   ├── main.py             # FastAPI router, CORS setup, API routes & static mounting
│   ├── model_utils.py      # Face cropping, MobileFaceNet embedding, similarity & validation
│   ├── dataset_prep.py     # DuckDuckGo scraper & curated fallback url downloads
│   ├── requirements.txt    # Python dependencies (TensorFlow, OpenCV, Supabase, etc.)
│   └── sessions/           # Local folder mapping cached session assets
└── frontend/
    ├── index.html          # SPA dashboard markup
    ├── style.css           # Glassmorphism CSS rules
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

### 2. Execution Commands
Run the backend server. The frontend folder is automatically served statically at `http://127.0.0.1:8000/`.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
