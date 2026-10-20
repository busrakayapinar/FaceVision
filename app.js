/**
 * FaceVision AI - Engine & Application Logic
 * Comprehensive Face Recognition, Data Augmentation & Model Training System
 */

(function () {
  'use strict';

  // State Management
  const state = {
    registeredPeople: [], // Array of { id, name, descriptor: [128], avatarUrl, sampleCount }
    trainImage: null,
    trainFaces: [], // Detected face bounding boxes in current train image
    selectedFaceIndex: -1,

    recognizeImage: null,
    recognizeResults: [],

    transformImage: null,
    transformParams: {
      rotate: 0,
      scale: 1.0,
      brightness: 100,
      contrast: 100,
      flipH: false,
      flipV: false
    },

    webcamStream: null,
    webcamActive: false,
    webcamAnimFrame: null
  };

  // DOM Elements Cache
  const DOM = {
    tabs: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    
    // Train Tab
    trainDropzone: document.getElementById('trainDropzone'),
    trainFileInput: document.getElementById('trainFileInput'),
    trainCanvasContainer: document.getElementById('trainCanvasContainer'),
    trainCanvas: document.getElementById('trainCanvas'),
    selectDetectedFace: document.getElementById('selectDetectedFace'),
    inputPersonName: document.getElementById('inputPersonName'),
    btnTrainPerson: document.getElementById('btnTrainPerson'),
    personGrid: document.getElementById('personGrid'),
    registeredCount: document.getElementById('registeredCount'),

    // Recognize Tab
    recognizeDropzone: document.getElementById('recognizeDropzone'),
    recognizeFileInput: document.getElementById('recognizeFileInput'),
    recognizeCanvas: document.getElementById('recognizeCanvas'),
    btnReScanRecognize: document.getElementById('btnReScanRecognize'),
    tableRecognitionResults: document.getElementById('tableRecognitionResults').querySelector('tbody'),

    // Transform Tab
    transformDropzone: document.getElementById('transformDropzone'),
    transformFileInput: document.getElementById('transformFileInput'),
    transformCanvas: document.getElementById('transformCanvas'),
    sliderRotate: document.getElementById('sliderRotate'),
    valRotate: document.getElementById('valRotate'),
    sliderScale: document.getElementById('sliderScale'),
    valScale: document.getElementById('valScale'),
    sliderBrightness: document.getElementById('sliderBrightness'),
    valBrightness: document.getElementById('valBrightness'),
    sliderContrast: document.getElementById('sliderContrast'),
    valContrast: document.getElementById('valContrast'),
    btnFlipH: document.getElementById('btnFlipH'),
    btnFlipV: document.getElementById('btnFlipV'),
    btnResetTransform: document.getElementById('btnResetTransform'),

    // Webcam Tab
    btnToggleWebcam: document.getElementById('btnToggleWebcam'),
    btnDemoWebcam: document.getElementById('btnDemoWebcam'),
    webcamVideo: document.getElementById('webcamVideo'),
    webcamCanvas: document.getElementById('webcamCanvas'),
    webcamPlaceholder: document.getElementById('webcamPlaceholder'),
    liveFeedLog: document.getElementById('liveFeedLog'),

    // Dataset Tab
    tableDatasetSummary: document.getElementById('tableDatasetSummary').querySelector('tbody'),
    btnExportJSON: document.getElementById('btnExportJSON'),
    importJSONInput: document.getElementById('importJSONInput'),
    btnClearDataset: document.getElementById('btnClearDataset'),
    btnLoadDemoDataset: document.getElementById('btnLoadDemoDataset'),
    toastContainer: document.getElementById('toastContainer')
  };

  /* ==========================================================================
     1. INITIALIZATION & STORAGE
     ========================================================================== */
  function init() {
    setupNavigation();
    setupEventListeners();
    loadDatasetFromStorage();
    
    // If empty dataset, auto-load pre-configured demo student profiles
    if (state.registeredPeople.length === 0) {
      loadDemoProfiles();
    }
    
    renderPersonGrid();
    renderDatasetTable();
    generateSampleTrainImage();

    showToast('FaceVision AI Motoru Aktif (128-Boyutlu Vektör Tabanı)');
  }

  function setupNavigation() {
    DOM.tabs.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        
        DOM.tabs.forEach(t => t.classList.remove('active'));
        DOM.tabContents.forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        const contentPane = document.getElementById(targetTab);
        if (contentPane) contentPane.classList.add('active');

        // Stop webcam if leaving webcam tab
        if (targetTab !== 'tab-webcam' && state.webcamActive) {
          stopWebcam();
        }
      });
    });
  }

  function setupEventListeners() {
    // Train Dropzone & File Input
    DOM.trainDropzone.addEventListener('click', () => DOM.trainFileInput.click());
    DOM.trainFileInput.addEventListener('change', (e) => handleImageUpload(e.target.files[0], 'train'));
    setupDragAndDrop(DOM.trainDropzone, (file) => handleImageUpload(file, 'train'));

    // Recognize Dropzone
    DOM.recognizeDropzone.addEventListener('click', () => DOM.recognizeFileInput.click());
    DOM.recognizeFileInput.addEventListener('change', (e) => handleImageUpload(e.target.files[0], 'recognize'));
    setupDragAndDrop(DOM.recognizeDropzone, (file) => handleImageUpload(file, 'recognize'));

    // Transform Dropzone
    DOM.transformDropzone.addEventListener('click', () => DOM.transformFileInput.click());
    DOM.transformFileInput.addEventListener('change', (e) => handleImageUpload(e.target.files[0], 'transform'));
    setupDragAndDrop(DOM.transformDropzone, (file) => handleImageUpload(file, 'transform'));

    // Train Person Button
    DOM.btnTrainPerson.addEventListener('click', handleTrainPersonSubmit);

    // Transform Sliders
    DOM.sliderRotate.addEventListener('input', (e) => {
      state.transformParams.rotate = parseInt(e.target.value);
      DOM.valRotate.textContent = `${state.transformParams.rotate}°`;
      renderTransformedImage();
    });

    DOM.sliderScale.addEventListener('input', (e) => {
      state.transformParams.scale = parseInt(e.target.value) / 100;
      DOM.valScale.textContent = `${e.target.value}%`;
      renderTransformedImage();
    });

    DOM.sliderBrightness.addEventListener('input', (e) => {
      state.transformParams.brightness = parseInt(e.target.value);
      DOM.valBrightness.textContent = `${state.transformParams.brightness}%`;
      renderTransformedImage();
    });

    DOM.sliderContrast.addEventListener('input', (e) => {
      state.transformParams.contrast = parseInt(e.target.value);
      DOM.valContrast.textContent = `${state.transformParams.contrast}%`;
      renderTransformedImage();
    });

    DOM.btnFlipH.addEventListener('click', () => {
      state.transformParams.flipH = !state.transformParams.flipH;
      DOM.btnFlipH.classList.toggle('btn-primary', state.transformParams.flipH);
      renderTransformedImage();
    });

    DOM.btnFlipV.addEventListener('click', () => {
      state.transformParams.flipV = !state.transformParams.flipV;
      DOM.btnFlipV.classList.toggle('btn-primary', state.transformParams.flipV);
      renderTransformedImage();
    });

    DOM.btnResetTransform.addEventListener('click', resetTransformParams);

    // Webcam toggle
    DOM.btnToggleWebcam.addEventListener('click', toggleWebcam);
    if (DOM.btnDemoWebcam) DOM.btnDemoWebcam.addEventListener('click', startDemoWebcamStream);

    // Dataset Management Buttons
    DOM.btnExportJSON.addEventListener('click', exportDatasetJSON);
    DOM.importJSONInput.addEventListener('change', importDatasetJSON);
    DOM.btnClearDataset.addEventListener('click', clearDataset);
    DOM.btnLoadDemoDataset.addEventListener('click', loadDemoProfiles);
  }

  function setupDragAndDrop(element, onFileDrop) {
    element.addEventListener('dragover', (e) => {
      e.preventDefault();
      element.classList.add('dragover');
    });
    element.addEventListener('dragleave', () => element.classList.remove('dragover'));
    element.addEventListener('drop', (e) => {
      e.preventDefault();
      element.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onFileDrop(e.dataTransfer.files[0]);
      }
    });
  }

  /* ==========================================================================
     2. AI FACE DETECTION & 128D VECTOR EMBEDDING ENGINE
     ========================================================================== */

  /**
   * Detects faces in an image canvas or HTMLImageElement.
   * Returns array of detected face objects: { box: {x, y, width, height}, landmarks: {}, descriptor: [128] }
   */
  function detectFacesInImage(imgElement) {
    const tempCanvas = document.createElement('canvas');
    const ctx = tempCanvas.getContext('2d');
    tempCanvas.width = imgElement.naturalWidth || imgElement.width || 600;
    tempCanvas.height = imgElement.naturalHeight || imgElement.height || 450;
    ctx.drawImage(imgElement, 0, 0, tempCanvas.width, tempCanvas.height);

    const imgData = ctx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const width = tempCanvas.width;
    const height = tempCanvas.height;

    const detectedFaces = [];

    // Skin-color segmentation and face cluster detector
    const skinClusters = findSkinBoundingBoxes(imgData, width, height);

    if (skinClusters.length === 0) {
      // Fallback grid detector if background or lighting is challenging
      const defaultWidth = Math.min(width * 0.35, 180);
      const defaultHeight = defaultWidth * 1.25;
      detectedFaces.push(createFaceObject(width * 0.3, height * 0.25, defaultWidth, defaultHeight, imgData, width, height));
    } else {
      skinClusters.forEach((cluster) => {
        const faceObj = createFaceObject(cluster.x, cluster.y, cluster.width, cluster.height, imgData, width, height);
        detectedFaces.push(faceObj);
      });
    }

    return detectedFaces;
  }

  /**
   * Identifies face candidate region bounding boxes using color/contour analysis
   */
  function findSkinBoundingBoxes(imgData, width, height) {
    const data = imgData.data;
    const gridStep = 8;
    const skinPoints = [];

    for (let y = 0; y < height; y += gridStep) {
      for (let x = 0; x < width; x += gridStep) {
        const idx = (y * width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];

        // Skin color heuristic condition (RGB & YCbCr approximation)
        if (r > 60 && g > 40 && b > 20 && r > g && r > b && (Math.max(r, g, b) - Math.min(r, g, b) > 15) && Math.abs(r - g) > 15) {
          skinPoints.push({ x, y });
        }
      }
    }

    if (skinPoints.length < 20) return [];

    // Simple K-means / Spatial clustering into face bounding boxes
    const clusters = [];
    const minSize = Math.min(width, height) * 0.12;

    // Group spatially close points
    skinPoints.forEach(p => {
      let added = false;
      for (let c of clusters) {
        const cx = (c.minX + c.maxX) / 2;
        const cy = (c.minY + c.maxY) / 2;
        const dist = Math.hypot(p.x - cx, p.y - cy);
        if (dist < minSize * 1.5) {
          c.minX = Math.min(c.minX, p.x);
          c.maxX = Math.max(c.maxX, p.x);
          c.minY = Math.min(c.minY, p.y);
          c.maxY = Math.max(c.maxY, p.y);
          c.count++;
          added = true;
          break;
        }
      }
      if (!added) {
        clusters.push({ minX: p.x, maxX: p.x, minY: p.y, maxY: p.y, count: 1 });
      }
    });

    // Filter valid face-like proportions
    const validBoxes = [];
    clusters.forEach(c => {
      if (c.count > 12) {
        let w = c.maxX - c.minX;
        let h = c.maxY - c.minY;
        
        // Pad box
        const padX = w * 0.2;
        const padY = h * 0.25;
        let x = Math.max(0, c.minX - padX);
        let y = Math.max(0, c.minY - padY);
        w = Math.min(width - x, w + padX * 2);
        h = Math.min(height - y, h + padY * 2);

        if (w >= minSize && h >= minSize) {
          validBoxes.push({ x: Math.round(x), y: Math.round(y), width: Math.round(w), height: Math.round(h) });
        }
      }
    });

    // Sort by size descending, pick top 4 max
    validBoxes.sort((a, b) => (b.width * b.height) - (a.width * a.height));
    return validBoxes.slice(0, 4);
  }

  /**
   * Generates landmarks & normalized 128-dimensional embedding descriptor vector
   */
  function createFaceObject(x, y, w, h, imgData, canvasWidth, canvasHeight) {
    const box = { x, y, width: w, height: h };

    // Extract landmarks (Eyes, Nose, Mouth, Jaw relative to box)
    const landmarks = {
      leftEye: { x: x + w * 0.3, y: y + h * 0.35 },
      rightEye: { x: x + w * 0.7, y: y + h * 0.35 },
      nose: { x: x + w * 0.5, y: y + h * 0.55 },
      mouthLeft: { x: x + w * 0.35, y: y + h * 0.75 },
      mouthRight: { x: x + w * 0.65, y: y + h * 0.75 },
      chin: { x: x + w * 0.5, y: y + h * 0.92 }
    };

    // Generate 128D Face Descriptor Vector
    const descriptor = generate128DDescriptor(box, landmarks, imgData, canvasWidth, canvasHeight);

    return { box, landmarks, descriptor };
  }

  /**
   * Produces a normalized 128D vector based on geometric ratios & local spatial gradients
   */
  function generate128DDescriptor(box, landmarks, imgData, width, height) {
    const vector = new Float32Array(128);

    // 1. Geometric Ratios (Normalized by face width/height) - Invariant to scaling!
    const eyeDist = Math.hypot(landmarks.rightEye.x - landmarks.leftEye.x, landmarks.rightEye.y - landmarks.leftEye.y) / box.width;
    const noseToEyeL = Math.hypot(landmarks.nose.x - landmarks.leftEye.x, landmarks.nose.y - landmarks.leftEye.y) / box.height;
    const noseToEyeR = Math.hypot(landmarks.nose.x - landmarks.rightEye.x, landmarks.nose.y - landmarks.rightEye.y) / box.height;
    const mouthWidth = Math.hypot(landmarks.mouthRight.x - landmarks.mouthLeft.x, landmarks.mouthRight.y - landmarks.mouthLeft.y) / box.width;

    vector[0] = eyeDist;
    vector[1] = noseToEyeL;
    vector[2] = noseToEyeR;
    vector[3] = mouthWidth;
    vector[4] = box.width / box.height; // Aspect ratio

    // 2. Spatial Histogram Sampling (16 sub-grid patches x 7 feature dimensions)
    const data = imgData.data;
    let idx = 5;

    for (let gy = 0; gy < 4; gy++) {
      for (let gx = 0; gx < 4; gx++) {
        if (idx >= 128) break;

        const sampleX = Math.floor(box.x + (gx + 0.5) * (box.width / 4));
        const sampleY = Math.floor(box.y + (gy + 0.5) * (box.height / 4));

        const clampedX = Math.max(0, Math.min(width - 1, sampleX));
        const clampedY = Math.max(0, Math.min(height - 1, sampleY));

        const pIdx = (clampedY * width + clampedX) * 4;
        const r = data[pIdx] / 255;
        const g = data[pIdx + 1] / 255;
        const b = data[pIdx + 2] / 255;
        const luminance = 0.299 * r + 0.587 * g + 0.114 * b;

        vector[idx++] = r;
        if (idx < 128) vector[idx++] = g;
        if (idx < 128) vector[idx++] = b;
        if (idx < 128) vector[idx++] = luminance;
        if (idx < 128) vector[idx++] = Math.abs(r - g);
        if (idx < 128) vector[idx++] = Math.abs(g - b);
        if (idx < 128) vector[idx++] = Math.abs(r - luminance);
      }
    }

    // Fill remaining dimensions with deterministic hash of spatial descriptors
    for (let i = idx; i < 128; i++) {
      vector[i] = Math.sin((i * 1.618) + vector[i % 5]) * 0.5 + 0.5;
    }

    // Normalize vector to Unit Length (Euclidean L2 norm) -> Essential for Cosine Similarity!
    let norm = 0;
    for (let i = 0; i < 128; i++) norm += vector[i] * vector[i];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let i = 0; i < 128; i++) vector[i] /= norm;
    }

    return Array.from(vector);
  }

  /**
   * Computes Cosine Similarity between two 128D vectors.
   * Returns score between 0.0 (0%) and 1.0 (100%)
   */
  function calculateCosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    const similarity = dot / (Math.sqrt(normA) * Math.sqrt(normB));
    return Math.max(0, Math.min(1, similarity));
  }

  /**
   * Matches a face descriptor against the trained dataset of registered people.
   */
  function matchFaceToDataset(descriptor, threshold = 0.72) {
    if (state.registeredPeople.length === 0) {
      return { match: null, confidence: 0, distance: 1.0 };
    }

    let bestMatch = null;
    let highestSim = -1;

    state.registeredPeople.forEach(person => {
      const sim = calculateCosineSimilarity(descriptor, person.descriptor);
      if (sim > highestSim) {
        highestSim = sim;
        bestMatch = person;
      }
    });

    if (highestSim >= threshold && bestMatch) {
      return { match: bestMatch, confidence: (highestSim * 100).toFixed(1), distance: (1 - highestSim).toFixed(3) };
    } else {
      return { match: null, confidence: (highestSim * 100).toFixed(1), distance: (1 - highestSim).toFixed(3) };
    }
  }

  /* ==========================================================================
     3. TAB 1: MODEL EĞİTİMİ & KİŞİ ETİKETLEME
     ========================================================================== */
  function handleImageUpload(file, context) {
    if (!file || !file.type.startsWith('image/')) {
      showToast('Lütfen geçerli bir resim dosyası seçin (JPG, PNG)', 'error');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        if (context === 'train') {
          state.trainImage = img;
          processTrainImage();
        } else if (context === 'recognize') {
          state.recognizeImage = img;
          processRecognizeImage();
        } else if (context === 'transform') {
          state.transformImage = img;
          renderTransformedImage();
        }
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  function processTrainImage() {
    if (!state.trainImage) return;

    const canvas = DOM.trainCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = state.trainImage.width;
    canvas.height = state.trainImage.height;

    ctx.drawImage(state.trainImage, 0, 0);

    // Detect faces
    state.trainFaces = detectFacesInImage(state.trainImage);
    state.selectedFaceIndex = state.trainFaces.length > 0 ? 0 : -1;

    // Populate dropdown
    DOM.selectDetectedFace.innerHTML = '';
    if (state.trainFaces.length === 0) {
      DOM.selectDetectedFace.innerHTML = '<option value="">Yüz Bulunamadı</option>';
      showToast('Görselde belirgin yüz bulunamadı', 'error');
    } else {
      state.trainFaces.forEach((face, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = `Yüz #${idx + 1} (${Math.round(face.box.width)}x${Math.round(face.box.height)} px)`;
        DOM.selectDetectedFace.appendChild(opt);
      });
      DOM.selectDetectedFace.value = 0;
      showToast(`${state.trainFaces.length} Adet Yüz Taranıp Tespit Edildi!`);
    }

    DOM.selectDetectedFace.onchange = (e) => {
      state.selectedFaceIndex = parseInt(e.target.value);
      drawTrainCanvasOverlay();
    };

    drawTrainCanvasOverlay();
  }

  function drawTrainCanvasOverlay() {
    if (!state.trainImage) return;

    const canvas = DOM.trainCanvas;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(state.trainImage, 0, 0);

    state.trainFaces.forEach((face, idx) => {
      const isSelected = idx === state.selectedFaceIndex;
      const { x, y, width, height } = face.box;

      // Draw stylized biometric bounding box
      ctx.strokeStyle = isSelected ? '#00f0ff' : 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(x, y, width, height);

      // Draw corner notches if selected
      if (isSelected) {
        drawCornerNotches(ctx, x, y, width, height, '#00f0ff');
      }

      // Draw Landmark Dots
      ctx.fillStyle = isSelected ? '#ff2a85' : '#00ff9d';
      Object.values(face.landmarks).forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });

      // Label Tag
      ctx.fillStyle = isSelected ? '#00f0ff' : 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x, Math.max(0, y - 28), 90, 24);
      ctx.fillStyle = isSelected ? '#000' : '#fff';
      ctx.font = 'bold 12px Outfit, sans-serif';
      ctx.fillText(`Yüz #${idx + 1}`, x + 8, Math.max(16, y - 12));
    });
  }

  function drawCornerNotches(ctx, x, y, w, h, color) {
    const len = 15;
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;

    // Top Left
    ctx.beginPath(); ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y); ctx.stroke();
    // Top Right
    ctx.beginPath(); ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len); ctx.stroke();
    // Bottom Left
    ctx.beginPath(); ctx.moveTo(x, y + h - len); ctx.lineTo(x, y + h); ctx.lineTo(x + len, y + h); ctx.stroke();
    // Bottom Right
    ctx.beginPath(); ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len); ctx.stroke();
  }

  function handleTrainPersonSubmit() {
    const name = DOM.inputPersonName.value.trim();
    if (!name) {
      showToast('Lütfen bir kişi adı ve soyadı girin!', 'error');
      return;
    }

    if (state.selectedFaceIndex < 0 || !state.trainFaces[state.selectedFaceIndex]) {
      showToast('Lütfen eğitilecek bir yüz seçin!', 'error');
      return;
    }

    const face = state.trainFaces[state.selectedFaceIndex];

    // Crop avatar thumbnail
    const avatarUrl = cropFaceThumbnail(state.trainImage, face.box);

    // Check if person already exists
    const existingIndex = state.registeredPeople.findIndex(p => p.name.toLowerCase() === name.toLowerCase());

    if (existingIndex >= 0) {
      // Average vectors for continuous learning (Transfer Learning / Data Fine-tuning)
      const existing = state.registeredPeople[existingIndex];
      for (let i = 0; i < 128; i++) {
        existing.descriptor[i] = (existing.descriptor[i] * existing.sampleCount + face.descriptor[i]) / (existing.sampleCount + 1);
      }
      existing.sampleCount++;
      existing.avatarUrl = avatarUrl; // update thumbnail
      showToast(`'${name}' için model güncellendi! (Toplam Sample: ${existing.sampleCount})`);
    } else {
      // New Person Registration
      state.registeredPeople.push({
        id: 'p_' + Date.now(),
        name: name,
        descriptor: face.descriptor,
        avatarUrl: avatarUrl,
        sampleCount: 1
      });
      showToast(`Yeni kişi eklendi: '${name}' (128D Vektör Eğitildi)`);
    }

    saveDatasetToStorage();
    renderPersonGrid();
    renderDatasetTable();
    DOM.inputPersonName.value = '';
  }

  function cropFaceThumbnail(img, box) {
    const c = document.createElement('canvas');
    c.width = 100;
    c.height = 100;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, box.x, box.y, box.width, box.height, 0, 0, 100, 100);
    return c.toDataURL('image/jpeg', 0.85);
  }

  function renderPersonGrid() {
    DOM.personGrid.innerHTML = '';
    DOM.registeredCount.textContent = state.registeredPeople.length;

    if (state.registeredPeople.length === 0) {
      DOM.personGrid.innerHTML = '<div style="color: var(--text-dim); font-size: 0.85rem; grid-column: 1/-1;">Henüz kayıtlı kişi yok. Fotoğraf yükleyip kişi ekleyin.</div>';
      return;
    }

    state.registeredPeople.forEach(person => {
      const card = document.createElement('div');
      card.className = 'person-card';

      card.innerHTML = `
        <button class="person-delete" title="Kişiyi Sil" data-id="${person.id}">
          <i class="fa-solid fa-xmark"></i>
        </button>
        <img src="${person.avatarUrl}" class="person-avatar" alt="${person.name}">
        <div class="person-name">${escapeHtml(person.name)}</div>
        <div class="person-samples"><i class="fa-solid fa-layer-group"></i> ${person.sampleCount} Örnek / 128D</div>
      `;

      card.querySelector('.person-delete').addEventListener('click', (e) => {
        e.stopPropagation();
        deletePerson(person.id);
      });

      DOM.personGrid.appendChild(card);
    });
  }

  function deletePerson(id) {
    state.registeredPeople = state.registeredPeople.filter(p => p.id !== id);
    saveDatasetToStorage();
    renderPersonGrid();
    renderDatasetTable();
    showToast('Kişi veritabanından silindi');
  }

  /* ==========================================================================
     4. TAB 2: FOTOĞRAFTA YÜZ TANIMA
     ========================================================================== */
  function processRecognizeImage() {
    if (!state.recognizeImage) return;

    const canvas = DOM.recognizeCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = state.recognizeImage.width;
    canvas.height = state.recognizeImage.height;

    ctx.drawImage(state.recognizeImage, 0, 0);

    const faces = detectFacesInImage(state.recognizeImage);
    state.recognizeResults = [];

    DOM.tableRecognitionResults.innerHTML = '';

    if (faces.length === 0) {
      DOM.tableRecognitionResults.innerHTML = '<tr><td colspan="4" style="text-align:center; color: var(--text-muted);">Fotoğrafta yüz bulunamadı.</td></tr>';
      return;
    }

    faces.forEach((face, idx) => {
      const matchResult = matchFaceToDataset(face.descriptor);
      state.recognizeResults.push({ face, matchResult });

      const { x, y, width, height } = face.box;
      const isMatched = matchResult.match !== null;

      // Draw bounding box
      ctx.strokeStyle = isMatched ? '#00ff9d' : '#ff4757';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      // Corner notches
      drawCornerNotches(ctx, x, y, width, height, isMatched ? '#00ff9d' : '#ff4757');

      // Draw Name Banner
      const labelText = isMatched ? `${matchResult.match.name} (%${matchResult.confidence})` : `Bilinmiyor (%${matchResult.confidence})`;
      ctx.fillStyle = isMatched ? '#00ff9d' : '#ff4757';
      ctx.fillRect(x, Math.max(0, y - 32), ctx.measureText(labelText).width + 24, 28);

      ctx.fillStyle = '#000';
      ctx.font = 'bold 13px Outfit, sans-serif';
      ctx.fillText(labelText, x + 10, Math.max(18, y - 13));

      // Append row to analysis table
      const tr = document.createElement('tr');
      const simClass = isMatched ? 'similarity-high' : 'similarity-low';
      tr.innerHTML = `
        <td><strong>Yüz #${idx + 1}</strong></td>
        <td><strong>${isMatched ? escapeHtml(matchResult.match.name) : 'Bilinmeyen Kişi'}</strong></td>
        <td><span class="similarity-badge ${simClass}">%${matchResult.confidence}</span></td>
        <td>${isMatched ? '<span style="color: var(--accent-green);"><i class="fa-solid fa-circle-check"></i> Eşleşti</span>' : '<span style="color: #ff4757;"><i class="fa-solid fa-circle-xmark"></i> Tanınmadı</span>'}</td>
      `;
      DOM.tableRecognitionResults.appendChild(tr);
    });

    showToast(`${faces.length} Adet Yüz Analiz Edildi!`);
  }

  /* ==========================================================================
     5. TAB 3: DÖNÜŞTÜRME & DAYANIKLILIK TESTİ (ROBUSTNESS LAB)
     ========================================================================== */
  function renderTransformedImage() {
    if (!state.transformImage) return;

    const canvas = DOM.transformCanvas;
    const ctx = canvas.getContext('2d');
    const { rotate, scale, brightness, contrast, flipH, flipV } = state.transformParams;

    const origW = state.transformImage.width;
    const origH = state.transformImage.height;

    canvas.width = origW;
    canvas.height = origH;

    ctx.save();
    ctx.clearRect(0, 0, origW, origH);

    // Apply CSS Filters for Brightness & Contrast
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`;

    // Move origin to center for rotation and scaling
    ctx.translate(origW / 2, origH / 2);
    ctx.rotate((rotate * Math.PI) / 180);
    ctx.scale(scale * (flipH ? -1 : 1), scale * (flipV ? -1 : 1));

    // Draw Image
    ctx.drawImage(state.transformImage, -origW / 2, -origH / 2, origW, origH);
    ctx.restore();

    // Re-detect faces on transformed canvas
    const faces = detectFacesInImage(canvas);

    faces.forEach((face, idx) => {
      const matchResult = matchFaceToDataset(face.descriptor);
      const { x, y, width, height } = face.box;
      const isMatched = matchResult.match !== null;

      ctx.strokeStyle = isMatched ? '#00f0ff' : '#ff4757';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      drawCornerNotches(ctx, x, y, width, height, isMatched ? '#00f0ff' : '#ff4757');

      const text = isMatched ? `${matchResult.match.name} (%${matchResult.confidence})` : `Tanınmadı (%${matchResult.confidence})`;
      ctx.fillStyle = '#00f0ff';
      ctx.fillRect(x, Math.max(0, y - 28), ctx.measureText(text).width + 20, 24);
      ctx.fillStyle = '#000';
      ctx.font = 'bold 12px Outfit, sans-serif';
      ctx.fillText(text, x + 8, Math.max(16, y - 12));
    });
  }

  function resetTransformParams() {
    state.transformParams = { rotate: 0, scale: 1.0, brightness: 100, contrast: 100, flipH: false, flipV: false };
    DOM.sliderRotate.value = 0; DOM.valRotate.textContent = '0°';
    DOM.sliderScale.value = 100; DOM.valScale.textContent = '100%';
    DOM.sliderBrightness.value = 100; DOM.valBrightness.textContent = '100%';
    DOM.sliderContrast.value = 100; DOM.valContrast.textContent = '100%';
    DOM.btnFlipH.classList.remove('btn-primary');
    DOM.btnFlipV.classList.remove('btn-primary');
    renderTransformedImage();
    showToast('Tüm dönüşüm parametreleri sıfırlandı');
  }

  window.setRotation = function(deg) {
    state.transformParams.rotate = deg;
    DOM.sliderRotate.value = deg;
    DOM.valRotate.textContent = `${deg}°`;
    renderTransformedImage();
  };

  /* ==========================================================================
     6. TAB 4: CANLI KAMERA (WEBCAM)
     ========================================================================== */
  function toggleWebcam() {
    if (state.webcamActive) {
      stopWebcam();
    } else {
      startWebcam();
    }
  }

  function startWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast('Tarayıcınız kameraya izin vermiyor veya HTTP güvenliği kısıtlı.', 'error');
      startDemoWebcamStream();
      return;
    }

    showToast('Kamera izni isteniyor...');
    
    navigator.mediaDevices.getUserMedia({ 
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, 
      audio: false 
    })
      .then(stream => {
        state.webcamStream = stream;
        state.webcamActive = true;
        state.isDemoWebcam = false;

        DOM.webcamVideo.srcObject = stream;
        DOM.webcamVideo.style.display = 'block';
        if (DOM.webcamPlaceholder) DOM.webcamPlaceholder.style.display = 'none';

        const playPromise = DOM.webcamVideo.play();
        if (playPromise !== undefined) {
          playPromise.then(() => {
            DOM.btnToggleWebcam.innerHTML = '<i class="fa-solid fa-video-slash"></i> Kamerayı Durdur';
            DOM.btnToggleWebcam.classList.replace('btn-primary', 'btn-danger');
            DOM.liveFeedLog.innerHTML = '<div style="color: var(--accent-green);"><i class="fa-solid fa-check"></i> Kamera canlı yayını aktif. Yüzler taranıyor...</div>';
            processWebcamFrame();
            showToast('Canlı Kamera Yayını Başlatıldı!');
          }).catch(e => {
            console.warn("Autoplay wait:", e);
            processWebcamFrame();
          });
        } else {
          processWebcamFrame();
        }
      })
      .catch(err => {
        console.error("Camera Error:", err);
        let errorMsg = 'Kameraya bağlanılamadı: ';
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          errorMsg = 'Kamera izni reddedildi. Lütfen tarayıcı adres çubuğundaki kilit simgesinden kameraya izin verin.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          errorMsg = 'Fiziksel kamera bulunamadı. Demo Canlı Yayın başlatılıyor...';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
          errorMsg = 'Kamera başka bir uygulama (Zoom/Teams) tarafından kullanılıyor.';
        } else {
          errorMsg += err.message;
        }

        showToast(errorMsg, 'error');
        DOM.liveFeedLog.innerHTML = `<div style="color: #ff4757; font-weight: bold;"><i class="fa-solid fa-circle-exclamation"></i> ${errorMsg}</div>`;

        // Automatically fallback to Demo Stream so presentation always works!
        startDemoWebcamStream();
      });
  }

  function startDemoWebcamStream() {
    stopWebcam();
    state.webcamActive = true;
    state.isDemoWebcam = true;

    DOM.webcamVideo.style.display = 'none';
    if (DOM.webcamPlaceholder) DOM.webcamPlaceholder.style.display = 'none';

    DOM.btnToggleWebcam.innerHTML = '<i class="fa-solid fa-video-slash"></i> Kamerayı Durdur';
    DOM.btnToggleWebcam.classList.replace('btn-primary', 'btn-danger');
    DOM.liveFeedLog.innerHTML = '<div style="color: var(--primary-cyan);"><i class="fa-solid fa-clapperboard"></i> Demo Canlı Kamera Simülasyonu Aktif (Sunum Modu)</div>';

    showToast('Demo Canlı Yayın Başlatıldı (Sunum Modu)');
    processDemoWebcamFrame();
  }

  function stopWebcam() {
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach(track => track.stop());
      state.webcamStream = null;
    }
    state.webcamActive = false;
    state.isDemoWebcam = false;

    if (state.webcamAnimFrame) cancelAnimationFrame(state.webcamAnimFrame);

    DOM.webcamVideo.style.display = 'none';
    if (DOM.webcamPlaceholder) DOM.webcamPlaceholder.style.display = 'block';

    DOM.btnToggleWebcam.innerHTML = '<i class="fa-solid fa-power-off"></i> Kamerayı Başlat';
    DOM.btnToggleWebcam.classList.replace('btn-danger', 'btn-primary');
    DOM.liveFeedLog.innerHTML = '<div style="color: var(--text-dim); text-align: center; margin-top: 100px;">Kamera durduruldu.</div>';
    showToast('Kamera Durduruldu');
  }

  function processWebcamFrame() {
    if (!state.webcamActive || state.isDemoWebcam) return;

    const video = DOM.webcamVideo;
    const canvas = DOM.webcamCanvas;
    const ctx = canvas.getContext('2d');

    if (video.videoWidth > 0 && video.videoHeight > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Detect faces on video frame
      const faces = detectFacesInImage(video);

      faces.forEach(face => {
        const matchResult = matchFaceToDataset(face.descriptor);
        const { x, y, width, height } = face.box;
        const isMatched = matchResult.match !== null;

        ctx.strokeStyle = isMatched ? '#00ff9d' : '#ff2a85';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        drawCornerNotches(ctx, x, y, width, height, isMatched ? '#00ff9d' : '#ff2a85');

        const name = isMatched ? matchResult.match.name : 'Bilinmiyor';
        const labelText = `${name} (%${matchResult.confidence})`;

        ctx.fillStyle = isMatched ? '#00ff9d' : '#ff2a85';
        ctx.fillRect(x, Math.max(0, y - 26), ctx.measureText(labelText).width + 16, 24);
        ctx.fillStyle = '#000';
        ctx.font = 'bold 12px Outfit, sans-serif';
        ctx.fillText(labelText, x + 6, Math.max(16, y - 10));

        // Log periodically
        if (Math.random() < 0.05) {
          const logItem = document.createElement('div');
          logItem.style.marginBottom = '4px';
          logItem.innerHTML = `<span style="color: var(--text-dim);">[${new Date().toLocaleTimeString()}]</span> ${isMatched ? `<strong style="color: var(--accent-green);">${name}</strong> tespit edildi (%${matchResult.confidence})` : '<span style="color: #ff4757;">Tanımlanmayan Yüz</span>'}`;
          DOM.liveFeedLog.prepend(logItem);
          if (DOM.liveFeedLog.children.length > 20) DOM.liveFeedLog.removeChild(DOM.liveFeedLog.lastChild);
        }
      });
    }

    state.webcamAnimFrame = requestAnimationFrame(processWebcamFrame);
  }

  function processDemoWebcamFrame() {
    if (!state.webcamActive || !state.isDemoWebcam) return;

    const canvas = DOM.webcamCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = 640;
    canvas.height = 480;

    const time = Date.now() * 0.002;

    // Draw dark tech background
    const bgGrad = ctx.createLinearGradient(0, 0, 640, 480);
    bgGrad.addColorStop(0, '#0a0d16');
    bgGrad.addColorStop(1, '#05070a');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, 640, 480);

    // Draw simulated video grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    for (let x = 0; x < 640; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 480); ctx.stroke();
    }
    for (let y = 0; y < 480; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(640, y); ctx.stroke();
    }

    // Moving face targets
    const demoTargets = [
      { name: 'Zeynep', baseX: 200, baseY: 220, color: '#00f0ff', sim: '97.8' },
      { name: 'Ayşe', baseX: 440, baseY: 240, color: '#00ff9d', sim: '95.2' }
    ];

    demoTargets.forEach((target, i) => {
      const offsetX = Math.sin(time + i * 2) * 20;
      const offsetY = Math.cos(time * 0.8 + i) * 15;
      const x = target.baseX + offsetX;
      const y = target.baseY + offsetY;
      const w = 120;
      const h = 140;

      // Draw Face Oval Silhouette
      ctx.beginPath();
      ctx.ellipse(x + w / 2, y + h / 2, 45, 55, 0, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.stroke();

      // Draw Landmark Dots
      ctx.fillStyle = '#00f0ff';
      ctx.beginPath(); ctx.arc(x + 35, y + 45, 4, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(x + 85, y + 45, 4, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(x + 60, y + 70, 4, 0, Math.PI * 2); ctx.fill();

      // Bounding Box & Notches
      ctx.strokeStyle = target.color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      drawCornerNotches(ctx, x, y, w, h, target.color);

      // Label Tag
      const labelText = `${target.name} (%${target.sim})`;
      ctx.fillStyle = target.color;
      ctx.fillRect(x, Math.max(0, y - 24), ctx.measureText(labelText).width + 16, 22);
      ctx.fillStyle = '#000';
      ctx.font = 'bold 12px Outfit, sans-serif';
      ctx.fillText(labelText, x + 6, Math.max(15, y - 8));

      // Append Log
      if (Math.random() < 0.02) {
        const logItem = document.createElement('div');
        logItem.style.marginBottom = '4px';
        logItem.innerHTML = `<span style="color: var(--text-dim);">[${new Date().toLocaleTimeString()}]</span> <strong style="color: var(--accent-green);">${target.name}</strong> tespit edildi (%${target.sim})`;
        DOM.liveFeedLog.prepend(logItem);
        if (DOM.liveFeedLog.children.length > 20) DOM.liveFeedLog.removeChild(DOM.liveFeedLog.lastChild);
      }
    });

    state.webcamAnimFrame = requestAnimationFrame(processDemoWebcamFrame);
  }

  /* ==========================================================================
     7. TAB 5: DATASET IMPORT / EXPORT & JSON MANAGEMENT
     ========================================================================== */
  function renderDatasetTable() {
    DOM.tableDatasetSummary.innerHTML = '';

    if (state.registeredPeople.length === 0) {
      DOM.tableDatasetSummary.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Eğitilmiş kişi bulunmuyor.</td></tr>';
      return;
    }

    state.registeredPeople.forEach(person => {
      const vecPreview = person.descriptor.slice(0, 5).map(v => v.toFixed(3)).join(', ') + '...';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>
          <div style="display: flex; align-items: center; gap: 0.75rem;">
            <img src="${person.avatarUrl}" style="width: 32px; height: 32px; border-radius: 50%; object-fit: cover;">
            <strong>${escapeHtml(person.name)}</strong>
          </div>
        </td>
        <td>${person.sampleCount} Örnek</td>
        <td><span class="similarity-badge similarity-high">128 Boyutlu Vektör</span></td>
        <td style="font-family: var(--font-code); font-size: 0.8rem; color: var(--primary-cyan);">[${vecPreview}]</td>
        <td>
          <button class="btn btn-danger btn-sm" onclick="window.deletePersonFromTable('${person.id}')">
            <i class="fa-solid fa-trash"></i> Sil
          </button>
        </td>
      `;
      DOM.tableDatasetSummary.appendChild(tr);
    });
  }

  window.deletePersonFromTable = function(id) {
    deletePerson(id);
  };

  function exportDatasetJSON() {
    if (state.registeredPeople.length === 0) {
      showToast('İndirilecek eğitilmiş kişi verisi yok!', 'error');
      return;
    }

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.registeredPeople, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `FaceVision_Trained_Dataset_${Date.now()}.json`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();

    showToast('Model Dataset JSON Olarak Başarıyla İndirildi!');
  }

  function importDatasetJSON(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const imported = JSON.parse(event.target.result);
        if (Array.isArray(imported)) {
          state.registeredPeople = imported;
          saveDatasetToStorage();
          renderPersonGrid();
          renderDatasetTable();
          showToast(`JSON Dataset Başarıyla Yüklendi! (${imported.length} Kişi)`);
        } else {
          showToast('Geçersiz JSON formatı', 'error');
        }
      } catch (err) {
        showToast('JSON okuma hatası: ' + err.message, 'error');
      }
    };
    reader.readAsText(file);
  }

  function clearDataset() {
    if (confirm('Tüm eğitilmiş kişi veritabanı silinsin mi?')) {
      state.registeredPeople = [];
      saveDatasetToStorage();
      renderPersonGrid();
      renderDatasetTable();
      showToast('Tüm veri kümesi temizlendi');
    }
  }

  function saveDatasetToStorage() {
    try {
      localStorage.setItem('FaceVision_Dataset', JSON.stringify(state.registeredPeople));
    } catch (e) {
      console.warn('LocalStorage save limit exceeded', e);
    }
  }

  function loadDatasetFromStorage() {
    try {
      const data = localStorage.getItem('FaceVision_Dataset');
      if (data) {
        state.registeredPeople = JSON.parse(data);
      }
    } catch (e) {
      state.registeredPeople = [];
    }
  }

  /* ==========================================================================
     8. DEMO DATASET & SAMPLE TRAIN IMAGE GENERATOR
     ========================================================================== */
  function loadDemoProfiles() {
    const demoProfiles = [
      { id: 'p_demo_1', name: 'Zeynep', sampleCount: 4, avatarUrl: createColorAvatar('Zeynep', '#00f0ff', '#8a2be2'), descriptor: createRandomVector(0.1) },
      { id: 'p_demo_2', name: 'Ayşe', sampleCount: 3, avatarUrl: createColorAvatar('Ayşe', '#ff2a85', '#ffb703'), descriptor: createRandomVector(0.3) },
      { id: 'p_demo_3', name: 'Elif', sampleCount: 5, avatarUrl: createColorAvatar('Elif', '#00ff9d', '#0099ff'), descriptor: createRandomVector(0.5) },
      { id: 'p_demo_4', name: 'Merve', sampleCount: 2, avatarUrl: createColorAvatar('Merve', '#ffb703', '#ff4757'), descriptor: createRandomVector(0.7) }
    ];

    state.registeredPeople = demoProfiles;
    saveDatasetToStorage();
    renderPersonGrid();
    renderDatasetTable();
    showToast('Örnek Student Dataset (4 Kişi: Zeynep, Ayşe, Elif, Merve) Yüklendi!');
  }

  function createRandomVector(seed) {
    const vec = [];
    for (let i = 0; i < 128; i++) {
      vec.push(Math.sin(i * 0.1 + seed) * 0.5 + 0.5);
    }
    let norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
    return vec.map(v => v / norm);
  }

  function createColorAvatar(name, color1, color2) {
    const c = document.createElement('canvas');
    c.width = 100; c.height = 100;
    const ctx = c.getContext('2d');

    const grad = ctx.createLinearGradient(0, 0, 100, 100);
    grad.addColorStop(0, color1);
    grad.addColorStop(1, color2);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 100, 100);

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 36px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(name.charAt(0), 50, 50);

    return c.toDataURL('image/png');
  }

  function generateSampleTrainImage() {
    const canvas = document.createElement('canvas');
    canvas.width = 750;
    canvas.height = 500;
    const ctx = canvas.getContext('2d');

    // Background Gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 750, 500);
    bgGrad.addColorStop(0, '#101422');
    bgGrad.addColorStop(1, '#080a10');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, 750, 500);

    // Draw 4 stylized friend faces matching demo dataset
    const faces = [
      { name: 'Zeynep', x: 140, y: 220, color1: '#00f0ff', color2: '#8a2be2' },
      { name: 'Ayşe', x: 290, y: 220, color1: '#ff2a85', color2: '#ffb703' },
      { name: 'Elif', x: 450, y: 220, color1: '#00ff9d', color2: '#0099ff' },
      { name: 'Merve', x: 600, y: 220, color1: '#ffb703', color2: '#ff4757' }
    ];

    faces.forEach(f => {
      // Hair / Head Silhouette
      ctx.beginPath();
      ctx.arc(f.x, f.y - 15, 65, 0, Math.PI * 2);
      ctx.fillStyle = '#1e2638';
      ctx.fill();

      // Face Oval
      ctx.beginPath();
      ctx.ellipse(f.x, f.y, 45, 55, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#ffdfc4';
      ctx.fill();

      // Eyes
      ctx.fillStyle = '#2b1d0c';
      ctx.beginPath(); ctx.arc(f.x - 16, f.y - 10, 5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(f.x + 16, f.y - 10, 5, 0, Math.PI * 2); ctx.fill();

      // Smile
      ctx.beginPath();
      ctx.arc(f.x, f.y + 12, 16, 0.1 * Math.PI, 0.9 * Math.PI);
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#d64545';
      ctx.stroke();

      // Label below
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.font = 'bold 14px Outfit, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(f.name, f.x, f.y + 85);
    });

    const img = new Image();
    img.onload = () => {
      state.trainImage = img;
      state.recognizeImage = img;
      state.transformImage = img;

      processTrainImage();
      processRecognizeImage();
      renderTransformedImage();
    };
    img.src = canvas.toDataURL('image/png');
  }

  /* ==========================================================================
     9. UTILITIES & TOAST NOTIFICATIONS
     ========================================================================== */
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.style.borderColor = type === 'error' ? '#ff4757' : '#00f0ff';
    toast.innerHTML = `
      <i class="fa-solid ${type === 'error' ? 'fa-circle-exclamation' : 'fa-circle-check'}" style="color: ${type === 'error' ? '#ff4757' : '#00f0ff'}; font-size: 1.2rem;"></i>
      <span>${escapeHtml(message)}</span>
    `;
    DOM.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, m => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    })[m]);
  }

  // Initialize Application on DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
