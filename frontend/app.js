
let trainingChart = null;
let pollInterval = null;
let isTrainingActive = false;
let activeTab = 'predict'; 

let currentSessionId = 'barca_players';
let sessionMetadata = null;
let sessionsList = [];

const dashboardGrid = document.getElementById('dashboard-grid');
const modelStatusBadge = document.getElementById('model-status-badge');

const btnNewSession = document.getElementById('btn-new-session');
const tabPredict = document.getElementById('tab-predict');
const tabModels = document.getElementById('tab-models');

const panelDatasetManager = document.getElementById('panel-dataset-manager');
const panelValidation = document.getElementById('panel-validation');
const panelInference = document.getElementById('panel-inference');
const panelModelManager = document.getElementById('panel-model-manager');

const sessionDisplayNameInput = document.getElementById('session-display-name');
const sessionClassesCountInput = document.getElementById('session-classes-count');
const epochsInput = document.getElementById('epochs');
const playerCardsContainer = document.getElementById('player-cards-container');
const btnTrainModel = document.getElementById('btn-train-model');
const btnRetrainRedirect = document.getElementById('btn-retrain-redirect');

const trainingConsole = document.getElementById('training-console');
const statusPulseDot = document.getElementById('status-pulse-dot');
const trainingStatusText = document.getElementById('training-status-text');
const trainingProgressFill = document.getElementById('training-progress-fill');
const metricEpoch = document.getElementById('metric-epoch');
const metricLoss = document.getElementById('metric-loss');
const metricAccuracy = document.getElementById('metric-accuracy');
const metricValAcc = document.getElementById('metric-val-acc');

const sandboxLockOverlay = document.getElementById('sandbox-lock-overlay');
const dropZone = document.getElementById('drop-zone');
const testImageUpload = document.getElementById('test-image-upload');
const previewImg = document.getElementById('preview-img');
const testImageUrl = document.getElementById('test-image-url');
const btnLoadUrl = document.getElementById('btn-load-url');
const resultsList = document.getElementById('results-list');
const activationGrid = document.getElementById('activation-grid');
const toastEl = document.getElementById('toast');
const croppedFaceContainer = document.getElementById('cropped-face-container');
const croppedFacePreview = document.getElementById('cropped-face-preview');

const modelGrid = document.getElementById('model-grid');
const importSessionFile = document.getElementById('import-session-file');

const galleryModal = document.getElementById('gallery-modal');
const galleryTitle = document.getElementById('gallery-title');
const galleryGrid = document.getElementById('gallery-grid');
const btnCloseGallery = document.getElementById('btn-close-gallery');
const btnToggleBulkDelete = document.getElementById('btn-toggle-bulk-delete');
const btnBulkDelete = document.getElementById('btn-bulk-delete');

const lightboxModal = document.getElementById('lightbox-modal');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxCaption = document.getElementById('lightbox-caption');
const btnCloseLightbox = document.getElementById('btn-close-lightbox');
const btnPrevLightbox = document.getElementById('btn-prev-lightbox');
const btnNextLightbox = document.getElementById('btn-next-lightbox');
const btnDeleteLightboxImg = document.getElementById('btn-delete-lightbox-img');

let activeGalleryClass = '';
let activeGalleryImages = [];
let selectedGalleryImages = new Set();
let isBulkDeleteMode = false;
let activeLightboxIndex = -1;

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    setupEventListeners();
    loadActiveSessionAndInit();
});

function showToast(message, duration = 3000) {
    toastEl.textContent = message;
    toastEl.classList.add('show');
    setTimeout(() => {
        toastEl.classList.remove('show');
    }, duration);
}

function initChart() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    Chart.defaults.color = '#9ca3af';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#a50044', 
                    backgroundColor: 'rgba(165, 0, 68, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    yAxisID: 'y-loss',
                    tension: 0.3
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#a50044',
                    backgroundColor: 'rgba(165, 0, 68, 0.2)',
                    borderWidth: 2,
                    yAxisID: 'y-loss',
                    tension: 0.3
                },
                {
                    label: 'Train Acc',
                    data: [],
                    borderColor: '#004d98', 
                    backgroundColor: 'rgba(0, 77, 152, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    yAxisID: 'y-acc',
                    tension: 0.3
                },
                {
                    label: 'Val Acc',
                    data: [],
                    borderColor: '#edbb00', 
                    backgroundColor: 'rgba(237, 187, 0, 0.2)',
                    borderWidth: 2,
                    yAxisID: 'y-acc',
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 15, padding: 15 }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Fold' }
                },
                'y-loss': {
                    type: 'linear',
                    position: 'left',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Loss' },
                    min: 0
                },
                'y-acc': {
                    type: 'linear',
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Accuracy' },
                    min: 0,
                    max: 1.0,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            }
        }
    });
}

function setupEventListeners() {
    
    tabPredict.addEventListener('click', () => switchTab('predict'));
    tabModels.addEventListener('click', () => switchTab('models'));
    btnNewSession.addEventListener('click', handleCreateNewSession);

    sessionDisplayNameInput.addEventListener('blur', updateSessionConfig);
    sessionClassesCountInput.addEventListener('change', handleClassesCountChange);
    
    btnTrainModel.addEventListener('click', startModelTraining);
    btnRetrainRedirect.addEventListener('click', handleRetrainRedirect);

    dropZone.addEventListener('click', () => testImageUpload.click());
    testImageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleTestImage(file);
    });
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleTestImage(file);
    });
    
    btnLoadUrl.addEventListener('click', loadTestImageFromUrl);
    testImageUrl.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') loadTestImageFromUrl();
    });

    importSessionFile.addEventListener('change', handleImportSession);

    btnCloseGallery.addEventListener('click', () => galleryModal.classList.remove('show'));
    btnToggleBulkDelete.addEventListener('click', toggleBulkDeleteMode);
    btnBulkDelete.addEventListener('click', executeBulkDelete);

    btnCloseLightbox.addEventListener('click', () => lightboxModal.classList.remove('show'));
    btnPrevLightbox.addEventListener('click', prevLightboxImage);
    btnNextLightbox.addEventListener('click', nextLightboxImage);
    btnDeleteLightboxImg.addEventListener('click', deleteActiveLightboxImage);

    document.addEventListener('keydown', (e) => {
        if (!lightboxModal.classList.contains('show')) return;
        if (e.key === 'Escape') lightboxModal.classList.remove('show');
        else if (e.key === 'ArrowLeft') prevLightboxImage();
        else if (e.key === 'ArrowRight') nextLightboxImage();
    });
}

function switchTab(tabName) {
    activeTab = tabName;
    
    tabPredict.classList.remove('active');
    tabModels.classList.remove('active');
    
    dashboardGrid.classList.remove('mode-predict', 'mode-edit', 'mode-manager');
    
    if (tabName === 'predict') {
        tabPredict.classList.add('active');
        dashboardGrid.classList.add('mode-predict');
        btnRetrainRedirect.style.display = 'inline-flex';
        checkSessionTrainStatus();
    } else if (tabName === 'models') {
        tabModels.classList.add('active');
        dashboardGrid.classList.add('mode-manager');
        loadSessionsGrid();
    } else if (tabName === 'edit') {
        dashboardGrid.classList.add('mode-edit');
        btnRetrainRedirect.style.display = 'none';
        checkSessionTrainStatus();
    }
}

async function loadActiveSessionAndInit() {
    try {
        const res = await fetch('/api/dataset-info');
        const data = await res.json();
        
        currentSessionId = data.active_session_id;
        await loadSessionMetadata(currentSessionId);
        switchTab('predict');
        checkInitialStatus();
    } catch (e) {
        console.error("Error initializing session info:", e);
        showToast("Error connecting to server. Attempting default reload.");
    }
}

async function loadSessionMetadata(sessionId) {
    try {
        const res = await fetch(`/api/sessions`);
        const sessions = await res.json();
        sessionsList = sessions;
        
        sessionMetadata = sessions.find(s => s.id === sessionId);
        if (!sessionMetadata) {
            console.error(`Session metadata for ${sessionId} not found`);
            return;
        }

        currentSessionId = sessionId;

        const activeSessionIndicator = document.getElementById('active-session-name-indicator');
        if (activeSessionIndicator) {
            activeSessionIndicator.textContent = sessionMetadata.display_name;
        }

        sessionDisplayNameInput.value = sessionMetadata.display_name;
        sessionClassesCountInput.value = sessionMetadata.classes.length;
        
        renderClassCards(sessionMetadata.classes);
        
        await updateClassImageCounts();

        updateChartHistory(sessionMetadata.history);

        updateStatusBadge(sessionMetadata.status);

        if (sessionMetadata.status === 'completed') {
            trainingConsole.classList.remove('inactive');
            trainingStatusText.textContent = "Status: Completed";
            trainingProgressFill.style.width = '100%';
            
            const history = sessionMetadata.history || {};
            const lossLen = history.loss ? history.loss.length : 0;
            if (lossLen > 0) {
                metricEpoch.textContent = `Fold ${lossLen}`;
                metricLoss.textContent = history.loss[lossLen - 1].toFixed(4);
                metricAccuracy.textContent = (history.accuracy[lossLen - 1] * 100).toFixed(1) + '%';
                metricValAcc.textContent = history.val_accuracy && history.val_accuracy.length > 0
                    ? (history.val_accuracy[lossLen - 1] * 100).toFixed(1) + '%'
                    : '-';
            }
        } else {
            trainingConsole.classList.add('inactive');
            trainingProgressFill.style.width = '0%';
            metricEpoch.textContent = '-';
            metricLoss.textContent = '-';
            metricAccuracy.textContent = '-';
            metricValAcc.textContent = '-';
        }
        
    } catch (e) {
        console.error("Error loading session metadata:", e);
    }
}

function checkSessionTrainStatus() {
    if (!sessionMetadata) return;
    
    const isCompleted = (sessionMetadata.status === 'completed');
    
    if (isCompleted) {
        sandboxLockOverlay.style.display = 'none';
    } else {
        
        if (activeTab === 'edit') {
            sandboxLockOverlay.style.display = 'flex';
        } else {
            sandboxLockOverlay.style.display = 'none';
        }
    }
}

function updateStatusBadge(status) {
    if (status === 'completed') {
        modelStatusBadge.textContent = "Model Trained";
        modelStatusBadge.classList.add('trained');
    } else if (status === 'training') {
        modelStatusBadge.textContent = "Model Training";
        modelStatusBadge.classList.remove('trained');
    } else {
        modelStatusBadge.textContent = "Model Untrained";
        modelStatusBadge.classList.remove('trained');
    }
}

function renderClassCards(classes) {
    playerCardsContainer.innerHTML = '';
    
    classes.forEach((className, index) => {
        const key = className.toLowerCase().replace(/[^a-z0-9]/g, '_');
        const cardHtml = `
            <div class="player-card" data-class="${className}">
                <div class="card-glow"></div>
                <div class="player-details" style="width: 100%;">
                    <input type="text" class="class-name-input" data-index="${index}" value="${className}" placeholder="Class Name (e.g. Player ${index + 1})">
                    
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
                        <div class="image-count" style="font-size: 0.8rem; color: var(--text-secondary);">
                            <i class="fa-regular fa-image"></i> <span class="count-val" id="count-${key}">0</span> images
                        </div>
                        <button class="btn btn-primary btn-gallery" style="padding: 4px 8px; font-size: 0.75rem;" data-class="${className}">
                            <i class="fa-solid fa-images"></i> Gallery
                        </button>
                    </div>

                    <!-- Progress bar container -->
                    <div class="upload-progress-container" id="progress-container-${key}" style="display: none;">
                        <div class="upload-progress-text" id="progress-text-${key}">Uploading: 0 / 0</div>
                        <div class="upload-progress-bar">
                            <div class="upload-progress-bar-fill" id="progress-bar-fill-${key}" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="upload-btn-wrapper" style="margin-left: 15px;">
                    <label class="btn-upload" for="upload-${key}" style="padding: 6px 12px; border-radius: 8px; font-size: 0.8rem; border: 1px solid var(--border-color); display: flex; align-items: center; gap: 5px; cursor: pointer;">
                        <i class="fa-solid fa-plus"></i> Add
                    </label>
                    <input type="file" id="upload-${key}" accept="image/*" class="train-upload-input" data-class="${className}" multiple style="display: none;">
                </div>
            </div>
        `;
        playerCardsContainer.insertAdjacentHTML('beforeend', cardHtml);
    });

    document.querySelectorAll('.class-name-input').forEach(input => {
        input.addEventListener('blur', handleClassNameBlur);
    });

    document.querySelectorAll('.train-upload-input').forEach(input => {
        input.addEventListener('change', handleClassFileUpload);
    });

    document.querySelectorAll('.player-card').forEach(card => {
        const className = card.dataset.class;
        const key = className.toLowerCase().replace(/[^a-z0-9]/g, '_');
        
        card.addEventListener('dragover', (e) => {
            e.preventDefault();
            card.classList.add('dragover');
        });
        card.addEventListener('dragleave', () => {
            card.classList.remove('dragover');
        });
        card.addEventListener('drop', (e) => {
            e.preventDefault();
            card.classList.remove('dragover');
            
            const items = e.dataTransfer.items;
            if (items) {
                let filesList = [];
                
                const resolveEntry = (entry) => {
                    return new Promise((resolve) => {
                        if (entry.isFile) {
                            entry.file((file) => {
                                if (file.type.startsWith('image/') || /\.(jpg|jpeg|png)$/i.test(file.name)) {
                                    filesList.push(file);
                                }
                                resolve();
                            });
                        } else if (entry.isDirectory) {
                            const dirReader = entry.createReader();
                            readAllEntries(dirReader, async (entries) => {
                                for (const child of entries) {
                                    await resolveEntry(child);
                                }
                                resolve();
                            });
                        } else {
                            resolve();
                        }
                    });
                };

                const readAllEntries = (dirReader, callback) => {
                    let all = [];
                    const read = () => {
                        dirReader.readEntries((entries) => {
                            if (entries.length === 0) {
                                callback(all);
                            } else {
                                all = all.concat(entries);
                                read();
                            }
                        }, () => callback(all));
                    };
                    read();
                };

                const promises = [];
                for (let i = 0; i < items.length; i++) {
                    const item = items[i];
                    if (item.kind === 'file') {
                        const entry = item.webkitGetAsEntry();
                        if (entry) promises.push(resolveEntry(entry));
                    }
                }

                Promise.all(promises).then(() => {
                    if (filesList.length > 0) {
                        triggerBatchUpload(className, filesList);
                    }
                });
            } else {
                const files = e.dataTransfer.files;
                if (files && files.length > 0) {
                    const filesList = Array.from(files).filter(f => f.type.startsWith('image/') || /\.(jpg|jpeg|png)$/i.test(f.name));
                    triggerBatchUpload(className, filesList);
                }
            }
        });
    });

    document.querySelectorAll('.btn-gallery').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const className = e.currentTarget.dataset.class;
            openHeadshotGallery(className);
        });
    });
}

async function updateClassImageCounts() {
    if (!sessionMetadata) return;
    try {
        const info = {};
        for (const className of sessionMetadata.classes) {
            const res = await fetch(`/api/sessions/${currentSessionId}/dataset/${className}`);
            const files = await res.json();
            const key = className.toLowerCase().replace(/[^a-z0-9]/g, '_');
            const countEl = document.getElementById(`count-${key}`);
            if (countEl) {
                countEl.textContent = files.length;
            }
            info[className] = files.length;
        }
        
        const total = Object.values(info).reduce((a, b) => a + b, 0);
        if (total < sessionMetadata.classes.length) {
            btnTrainModel.disabled = true;
            btnTrainModel.title = "Upload at least 1 image per class before training.";
        } else {
            btnTrainModel.disabled = isTrainingActive;
            btnTrainModel.title = "";
        }
    } catch (e) {
        console.error("Error updating image counts:", e);
    }
}

async function updateSessionConfig() {
    const newName = sessionDisplayNameInput.value.trim();
    if (!newName || newName === sessionMetadata.display_name) return;
    
    try {
        const res = await fetch(`/api/sessions/${currentSessionId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ display_name: newName })
        });
        if (res.ok) {
            const data = await res.json();
            sessionMetadata.display_name = data.metadata.display_name;
            const activeSessionIndicator = document.getElementById('active-session-name-indicator');
            if (activeSessionIndicator) {
                activeSessionIndicator.textContent = newName;
            }
            showToast(`Session display name updated to: ${newName}`);
        } else {
            showToast("Failed to rename session display name.");
        }
    } catch (e) {
        console.error(e);
        showToast("Error renaming session display name.");
    }
}

async function handleClassesCountChange() {
    let count = parseInt(sessionClassesCountInput.value);
    if (isNaN(count) || count < 2) count = 2;
    if (count > 10) count = 10;
    sessionClassesCountInput.value = count;

    let currentClasses = [...sessionMetadata.classes];
    
    if (count > currentClasses.length) {
        
        while (currentClasses.length < count) {
            currentClasses.push(`Class ${currentClasses.length + 1}`);
        }
    } else if (count < currentClasses.length) {
        
        currentClasses = currentClasses.slice(0, count);
    }

    await saveSessionClasses(currentClasses);
}

async function saveSessionClasses(newClasses) {
    try {
        const res = await fetch(`/api/sessions/${currentSessionId}/classes`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ classes: newClasses })
        });
        if (res.ok) {
            const data = await res.json();
            sessionMetadata.classes = data.metadata.classes;
            renderClassCards(sessionMetadata.classes);
            await updateClassImageCounts();
            showToast("Session class list updated.");
        } else {
            showToast("Failed to update class list configuration.");
        }
    } catch (e) {
        console.error(e);
        showToast("Error updating class list configuration.");
    }
}

async function handleClassNameBlur(e) {
    const idx = parseInt(e.target.dataset.index);
    const newName = e.target.value.trim();
    if (!newName) {
        e.target.value = sessionMetadata.classes[idx]; 
        return;
    }
    
    if (newName === sessionMetadata.classes[idx]) return;
    
    const updatedClasses = [...sessionMetadata.classes];
    updatedClasses[idx] = newName;
    
    await saveSessionClasses(updatedClasses);
}

function handleClassFileUpload(e) {
    const className = e.target.dataset.class;
    const files = e.target.files;
    if (files && files.length > 0) {
        triggerBatchUpload(className, Array.from(files));
    }
    e.target.value = ''; 
}

async function triggerBatchUpload(className, filesList) {
    const key = className.toLowerCase().replace(/[^a-z0-9]/g, '_');
    const container = document.getElementById(`progress-container-${key}`);
    const textEl = document.getElementById(`progress-text-${key}`);
    const fillEl = document.getElementById(`progress-bar-fill-${key}`);

    container.style.display = 'block';
    fillEl.classList.remove('success-fill');
    textEl.classList.remove('success-text');

    const total = filesList.length;
    let completed = 0;
    
    textEl.textContent = `Uploading: 0 / ${total}`;
    fillEl.style.width = `0%`;

    const queue = [...filesList];
    const uploadNext = async () => {
        if (queue.length === 0) return;
        const file = queue.shift();
        
        const formData = new FormData();
        formData.append('player', className);
        formData.append('file', file);

        try {
            const res = await fetch(`/api/sessions/${currentSessionId}/upload`, {
                method: 'POST',
                body: formData
            });
            if (res.ok) {
                completed++;
                const pct = Math.round((completed / total) * 100);
                textEl.textContent = `Uploading: ${completed} / ${total}`;
                fillEl.style.width = `${pct}%`;
            }
        } catch (e) {
            console.error("Upload error:", e);
        }
        
        await uploadNext();
    };

    const workers = [uploadNext(), uploadNext(), uploadNext()];
    await Promise.all(workers);

    textEl.textContent = `Upload Finished (${total} images)`;
    fillEl.style.width = `100%`;
    fillEl.classList.add('success-fill');
    textEl.classList.add('success-text');

    showToast(`Successfully uploaded ${completed} images for ${className.toUpperCase()}`);
    await updateClassImageCounts();
}

async function startModelTraining() {
    if (isTrainingActive) return;
    
    const folds = epochsInput.value;
    
    btnTrainModel.disabled = true;
    btnTrainModel.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Validating...';
    
    try {
        const res = await fetch(`/api/sessions/${currentSessionId}/train?folds=${folds}`, { method: 'POST' });
        const data = await res.json();
        
        if (data.status === 'started') {
            showToast("Vector space cross-validation started.");
            isTrainingActive = true;
            trainingConsole.classList.remove('inactive');
            
            trainingChart.data.labels = [];
            trainingChart.data.datasets.forEach(d => d.data = []);
            trainingChart.update();
            
            startPollingStatus();
        } else {
            showToast(data.message || "Failed to start validation.");
            btnTrainModel.innerHTML = '<i class="fa-solid fa-gears"></i> Run Vector Space Cross-Validation';
            btnTrainModel.disabled = false;
        }
    } catch (e) {
        showToast("Error starting validation.");
        btnTrainModel.innerHTML = '<i class="fa-solid fa-gears"></i> Run Vector Space Cross-Validation';
        btnTrainModel.disabled = false;
        console.error(e);
    }
}

function startPollingStatus() {
    if (pollInterval) return; 
    
    pollInterval = setInterval(async () => {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            
            if (!data.training) {
                if (isTrainingActive) {
                    isTrainingActive = false;
                    btnTrainModel.disabled = false;
                    btnTrainModel.innerHTML = '<i class="fa-solid fa-gears"></i> Run Vector Space Cross-Validation';
                    
                    if (data.keras_logs && data.keras_logs.status === 'completed') {
                        showToast("K-Fold cross-validation completed! Templates cached.");
                        
                        await loadSessionMetadata(currentSessionId);
                        checkSessionTrainStatus();
                    } else if (data.keras_logs && data.keras_logs.status === 'failed') {
                        showToast("Validation failed: " + (data.keras_logs.error || "Unknown error"));
                        trainingStatusText.textContent = "Status: Failed";
                    }
                }
                clearInterval(pollInterval);
                pollInterval = null;
                return;
            }
            
            if (data.training) {
                isTrainingActive = true;
                btnTrainModel.disabled = true;
                btnTrainModel.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Validating...';
                trainingConsole.classList.remove('inactive');
                
                if (data.keras_logs) {
                    const logs = data.keras_logs;
                    const curEpoch = logs.current_epoch || 0;
                    const totEpochs = logs.total_epochs || 5;
                    const percent = Math.round((curEpoch / totEpochs) * 100);
                    
                    trainingStatusText.textContent = `Status: Validation (Fold ${curEpoch}/${totEpochs})`;
                    trainingProgressFill.style.width = `${percent}%`;
                    
                    const history = logs.history || {};
                    const lossLen = history.loss ? history.loss.length : 0;
                    
                    if (lossLen > 0) {
                        metricEpoch.textContent = `Fold ${curEpoch}`;
                        metricLoss.textContent = history.loss[lossLen - 1].toFixed(4);
                        metricAccuracy.textContent = (history.accuracy[lossLen - 1] * 100).toFixed(1) + '%';
                        metricValAcc.textContent = history.val_accuracy && history.val_accuracy.length > 0
                            ? (history.val_accuracy[lossLen - 1] * 100).toFixed(1) + '%'
                            : '-';
                            
                        updateChartHistory(history);
                    }
                }
            }
        } catch (e) {
            console.error("Error polling status:", e);
        }
    }, 1500);
}

function updateChartHistory(history) {
    if (!history || !history.loss) return;
    const lossLen = history.loss.length;
    const labels = Array.from({length: lossLen}, (_, i) => `Fold ${i + 1}`);
    trainingChart.data.labels = labels;
    trainingChart.data.datasets[0].data = history.loss;
    trainingChart.data.datasets[1].data = history.val_loss || [];
    trainingChart.data.datasets[2].data = history.accuracy;
    trainingChart.data.datasets[3].data = history.val_accuracy || [];
    trainingChart.update();
}

function handleRetrainRedirect() {
    switchTab('edit');
}

async function checkInitialStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.training) {
            startPollingStatus();
        }
    } catch (e) {
        console.error("Error checking initial status:", e);
    }
}

function handleTestImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
    };
    reader.readAsDataURL(file);
    predictImageFile(file);
}

function loadTestImageFromUrl() {
    const url = testImageUrl.value.trim();
    if (!url) return;
    previewImg.style.display = 'none';
    showToast("Attempting to load image from URL...");
    
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = function() {
        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this, 0, 0);
        canvas.toBlob((blob) => {
            const file = new File([blob], "url_image.jpg", { type: "image/jpeg" });
            previewImg.src = url;
            previewImg.style.display = 'block';
            predictImageFile(file);
        }, 'image/jpeg');
    };
    img.onerror = function() {
        showToast("CORS blocked direct load. Please download image and drop it!");
    };
    img.src = url;
}

async function predictImageFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    croppedFaceContainer.style.display = 'none';
    resultsList.innerHTML = '<p class="placeholder-text"><i class="fa-solid fa-spinner fa-spin"></i> Running CNN inference...</p>';
    activationGrid.innerHTML = `
        <div class="placeholder-maps">
            <i class="fa-solid fa-spinner fa-spin"></i>
            <p>Extracting convolutional activations...</p>
        </div>
    `;
    
    try {
        const res = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await res.json();
        
        if (res.ok) {
            if (data.cropped_face) {
                croppedFacePreview.src = 'data:image/jpeg;base64,' + data.cropped_face;
                croppedFaceContainer.style.display = 'block';
            } else {
                croppedFaceContainer.style.display = 'none';
            }
            renderPredictions(data.predictions);
            renderActivationMaps(data.activation_maps);
        } else {
            resultsList.innerHTML = `<p class="placeholder-text" style="color:#ef4444;"><i class="fa-solid fa-triangle-exclamation"></i> ${data.error || 'Prediction failed'}</p>`;
        }
    } catch (e) {
        resultsList.innerHTML = '<p class="placeholder-text" style="color:#ef4444;"><i class="fa-solid fa-triangle-exclamation"></i> Network error running prediction.</p>';
        console.error(e);
    }
}

function renderPredictions(predictions) {
    resultsList.innerHTML = '';
    if (!predictions || predictions.length === 0) {
        resultsList.innerHTML = '<p class="placeholder-text">No predictions found.</p>';
        return;
    }
    
    predictions.forEach(pred => {
        const percent = (pred.confidence * 100).toFixed(1);
        const itemHtml = `
            <div class="prediction-item">
                <div class="prediction-info-row">
                    <span class="player-name-label">${pred.class.toUpperCase()}</span>
                    <span class="prediction-conf-val">${percent}%</span>
                </div>
                <div class="prediction-bar-container">
                    <div class="prediction-bar-fill" style="width: ${percent}%; background: var(--barca-gold);"></div>
                </div>
            </div>
        `;
        resultsList.insertAdjacentHTML('beforeend', itemHtml);
    });
}

function renderActivationMaps(maps) {
    activationGrid.innerHTML = '';
    if (!maps || maps.length === 0) {
        activationGrid.innerHTML = '<div class="placeholder-maps"><p>No feature maps extracted.</p></div>';
        return;
    }
    maps.forEach((base64Str, idx) => {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${base64Str}`;
        img.className = 'activation-map-img';
        img.alt = `Filter ${idx + 1}`;
        activationGrid.appendChild(img);
    });
}

async function loadSessionsGrid() {
    modelGrid.innerHTML = '<div class="placeholder-text"><i class="fa-solid fa-spinner fa-spin"></i> Loading session profiles...</div>';
    
    try {
        const res = await fetch('/api/sessions');
        const sessions = await res.json();
        sessionsList = sessions;
        
        modelGrid.innerHTML = '';
        if (sessions.length === 0) {
            modelGrid.innerHTML = '<div class="placeholder-text">No sessions created yet. Click "+ New Session" to get started.</div>';
            return;
        }

        sessions.forEach(sess => {
            const date = new Date(sess.created_at).toLocaleDateString();
            const isActive = sess.is_active;
            const statusClass = sess.status;
            
            const cardHtml = `
                <div class="model-session-card ${isActive ? 'active-session' : ''}" data-id="${sess.id}">
                    ${isActive ? '<span class="active-badge"><i class="fa-solid fa-circle-check"></i> Active</span>' : ''}
                    <div class="model-card-header">
                        <div class="model-card-title">
                            <h3>${sess.display_name}</h3>
                            <span class="model-id">ID: ${sess.id}</span>
                        </div>
                        <span class="model-badge ${statusClass}">${statusClass}</span>
                    </div>
                    <div class="model-card-info">
                        <div>Classes count: <strong>${sess.classes.length}</strong></div>
                        <div>Classes list: <strong>${sess.classes.join(', ')}</strong></div>
                        <div>Created on: <strong>${date}</strong></div>
                    </div>
                    <div class="model-card-actions">
                        ${!isActive ? `<button class="btn btn-primary btn-activate-sess" data-id="${sess.id}"><i class="fa-solid fa-play"></i> Activate</button>` : ''}
                        <button class="btn btn-secondary btn-edit-sess" data-id="${sess.id}"><i class="fa-solid fa-pen-to-square"></i> Edit/Manage</button>
                        <button class="btn btn-secondary btn-export-sess" data-id="${sess.id}"><i class="fa-solid fa-file-zipper"></i> Export</button>
                        ${!isActive ? `<button class="btn btn-secondary btn-delete-sess" data-id="${sess.id}" style="border: 1px solid rgba(239, 68, 68, 0.4);"><i class="fa-solid fa-trash-can"></i> Delete</button>` : ''}
                    </div>
                </div>
            `;
            modelGrid.insertAdjacentHTML('beforeend', cardHtml);
        });

        document.querySelectorAll('.btn-activate-sess').forEach(btn => {
            btn.addEventListener('click', (e) => handleActivateSession(e.target.dataset.id));
        });
        document.querySelectorAll('.btn-edit-sess').forEach(btn => {
            btn.addEventListener('click', (e) => handleEditSession(e.target.dataset.id));
        });
        document.querySelectorAll('.btn-export-sess').forEach(btn => {
            btn.addEventListener('click', (e) => handleExportSession(e.target.dataset.id));
        });
        document.querySelectorAll('.btn-delete-sess').forEach(btn => {
            btn.addEventListener('click', (e) => handleDeleteSession(e.target.dataset.id));
        });
        
    } catch (e) {
        console.error(e);
        modelGrid.innerHTML = '<div class="placeholder-text" style="color:#ef4444;"><i class="fa-solid fa-triangle-exclamation"></i> Error loading sessions from server.</div>';
    }
}

async function handleActivateSession(sessionId) {
    showToast("Switching active session...");
    try {
        const res = await fetch(`/api/sessions/${sessionId}/activate`, { method: 'POST' });
        if (res.ok) {
            showToast("Session switched successfully!");
            await loadSessionMetadata(sessionId);
            switchTab('predict');
        } else {
            showToast("Failed to switch active session.");
        }
    } catch (e) {
        console.error(e);
        showToast("Error switching active session.");
    }
}

async function handleEditSession(sessionId) {
    showToast(`Loading details for: ${sessionId}`);
    await loadSessionMetadata(sessionId);
    switchTab('edit');
}

async function handleCreateNewSession() {
    showToast("Creating new session profile...");
    
    const timestamp = Date.now();
    const id = `session_${timestamp}`;
    const name = `New Session ${new Date().toLocaleDateString()}`;
    
    try {
        const res = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                id: id,
                display_name: name,
                classes: ["Class 1", "Class 2", "Class 3"]
            })
        });
        
        if (res.ok) {
            const data = await res.json();
            showToast("New session profile allocated. Switched to editor.");
            await loadSessionMetadata(data.id);
            switchTab('edit');
            
            setTimeout(() => {
                sessionDisplayNameInput.focus();
                sessionDisplayNameInput.select();
            }, 100);
        } else {
            const err = await res.json();
            showToast(`Failed to create session: ${err.detail || 'Unknown error'}`);
        }
    } catch (e) {
        console.error(e);
        showToast("Error creating session profile.");
    }
}

function handleExportSession(sessionId) {
    showToast("Preparing session zip archive...");
    window.location.href = `/api/sessions/${sessionId}/export`;
}

async function handleImportSession(e) {
    const file = e.target.files[0];
    if (!file) return;

    showToast("Uploading session archive...");
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/sessions/import', {
            method: 'POST',
            body: formData
        });
        if (res.ok) {
            const data = await res.json();
            showToast(`Successfully imported: ${data.metadata.display_name}`);
            loadSessionsGrid();
        } else {
            const err = await res.json();
            showToast(`Import failed: ${err.detail || 'Invalid archive format'}`);
        }
    } catch (e) {
        console.error(e);
        showToast("Error uploading import file.");
    }
    e.target.value = ''; 
}

async function handleDeleteSession(sessionId) {
    const confirmDel = confirm(`Are you sure you want to delete session profile '${sessionId}'? This deletes all templates and crop images folder!`);
    if (!confirmDel) return;

    showToast("Deleting session profile...");
    try {
        const res = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        if (res.ok) {
            showToast("Session deleted successfully.");
            loadSessionsGrid();
        } else {
            const err = await res.json();
            showToast(`Delete failed: ${err.detail || 'Access restricted'}`);
        }
    } catch (e) {
        console.error(e);
        showToast("Error deleting session profile.");
    }
}

async function openHeadshotGallery(className) {
    activeGalleryClass = className;
    galleryTitle.textContent = `Headshot Gallery: ${className.toUpperCase()}`;
    
    selectedGalleryImages.clear();
    isBulkDeleteMode = false;
    btnToggleBulkDelete.innerHTML = '<i class="fa-solid fa-square-check"></i> Select Images';
    btnBulkDelete.style.display = 'none';

    galleryModal.classList.add('show');
    await loadGalleryGrid();
}

async function loadGalleryGrid() {
    galleryGrid.innerHTML = '<div class="placeholder-text"><i class="fa-solid fa-spinner fa-spin"></i> Loading headshots...</div>';
    
    try {
        const res = await fetch(`/api/sessions/${currentSessionId}/dataset/${activeGalleryClass}`);
        const files = await res.json();
        activeGalleryImages = files;
        
        galleryGrid.innerHTML = '';
        if (files.length === 0) {
            galleryGrid.innerHTML = '<div class="placeholder-text" style="grid-column:1/-1;">No headshot crops uploaded for this class yet.</div>';
            return;
        }

        files.forEach((filename, idx) => {
            const imgUrl = `/api/sessions/${currentSessionId}/dataset/${activeGalleryClass}/${filename}`;
            const itemHtml = `
                <div class="gallery-item ${isBulkDeleteMode ? 'bulk-mode' : ''} ${selectedGalleryImages.has(filename) ? 'selected' : ''}" data-filename="${filename}" data-index="${idx}">
                    <img src="${imgUrl}" alt="Crop image">
                    <div class="gallery-checkbox-overlay"></div>
                </div>
            `;
            galleryGrid.insertAdjacentHTML('beforeend', itemHtml);
        });

        document.querySelectorAll('.gallery-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const filename = e.currentTarget.dataset.filename;
                const index = parseInt(e.currentTarget.dataset.index);
                
                if (isBulkDeleteMode) {
                    toggleSelectImage(e.currentTarget, filename);
                } else {
                    openLightbox(index);
                }
            });
        });

    } catch (e) {
        console.error(e);
        galleryGrid.innerHTML = '<div class="placeholder-text" style="grid-column:1/-1; color:#ef4444;">Error loading crops list.</div>';
    }
}

function toggleBulkDeleteMode() {
    isBulkDeleteMode = !isBulkDeleteMode;
    selectedGalleryImages.clear();
    btnBulkDelete.style.display = 'none';

    if (isBulkDeleteMode) {
        btnToggleBulkDelete.innerHTML = '<i class="fa-solid fa-xmark"></i> Cancel';
        document.querySelectorAll('.gallery-item').forEach(item => {
            item.classList.add('bulk-mode');
        });
    } else {
        btnToggleBulkDelete.innerHTML = '<i class="fa-solid fa-square-check"></i> Select Images';
        document.querySelectorAll('.gallery-item').forEach(item => {
            item.classList.remove('bulk-mode', 'selected');
        });
    }
}

function toggleSelectImage(element, filename) {
    if (selectedGalleryImages.has(filename)) {
        selectedGalleryImages.delete(filename);
        element.classList.remove('selected');
    } else {
        selectedGalleryImages.add(filename);
        element.classList.add('selected');
    }

    if (selectedGalleryImages.size > 0) {
        btnBulkDelete.style.display = 'inline-flex';
        btnBulkDelete.innerHTML = `<i class="fa-solid fa-trash-can"></i> Delete Selected (${selectedGalleryImages.size})`;
    } else {
        btnBulkDelete.style.display = 'none';
    }
}

async function executeBulkDelete() {
    const list = Array.from(selectedGalleryImages);
    const confirmDel = confirm(`Are you sure you want to bulk-delete ${list.length} headshots?`);
    if (!confirmDel) return;

    showToast("Deleting selected images...");
    try {
        const res = await fetch(`/api/sessions/${currentSessionId}/dataset/${activeGalleryClass}`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filenames: list })
        });
        
        if (res.ok) {
            showToast(`Deleted ${list.length} images successfully.`);
            toggleBulkDeleteMode();
            await loadGalleryGrid();
            await updateClassImageCounts();
        } else {
            showToast("Failed to delete selected images.");
        }
    } catch (e) {
        console.error(e);
        showToast("Error executing bulk deletion.");
    }
}

function openLightbox(index) {
    activeLightboxIndex = index;
    lightboxModal.classList.add('show');
    loadLightboxImage();
}

function loadLightboxImage() {
    if (activeLightboxIndex < 0 || activeLightboxIndex >= activeGalleryImages.length) return;
    const filename = activeGalleryImages[activeLightboxIndex];
    const imgUrl = `/api/sessions/${currentSessionId}/dataset/${activeGalleryClass}/${filename}`;
    
    lightboxImg.src = imgUrl;
    lightboxCaption.textContent = `Image ${activeLightboxIndex + 1} of ${activeGalleryImages.length}: ${filename}`;
}

function prevLightboxImage() {
    if (activeGalleryImages.length <= 1) return;
    activeLightboxIndex = (activeLightboxIndex - 1 + activeGalleryImages.length) % activeGalleryImages.length;
    loadLightboxImage();
}

function nextLightboxImage() {
    if (activeGalleryImages.length <= 1) return;
    activeLightboxIndex = (activeLightboxIndex + 1) % activeGalleryImages.length;
    loadLightboxImage();
}

async function deleteActiveLightboxImage() {
    if (activeLightboxIndex < 0) return;
    const filename = activeGalleryImages[activeLightboxIndex];
    const confirmDel = confirm("Are you sure you want to delete this image?");
    if (!confirmDel) return;

    try {
        const res = await fetch(`/api/sessions/${currentSessionId}/dataset/${activeGalleryClass}`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filenames: [filename] })
        });
        
        if (res.ok) {
            showToast("Image deleted successfully.");
            
            activeGalleryImages.splice(activeLightboxIndex, 1);
            await updateClassImageCounts();

            if (activeGalleryImages.length === 0) {
                
                lightboxModal.classList.remove('show');
                loadGalleryGrid();
            } else {
                
                if (activeLightboxIndex >= activeGalleryImages.length) {
                    activeLightboxIndex = activeGalleryImages.length - 1;
                }
                loadLightboxImage();
            }
        } else {
            showToast("Failed to delete image.");
        }
    } catch (e) {
        console.error(e);
        showToast("Error deleting image.");
    }
}
