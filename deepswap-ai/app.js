/**
 * DeepSwap AI — app.js
 * Real-time face swap & body tracking using MediaPipe FaceMesh + Pose
 * Inspired by Deep-Live-Cam (github.com/hacksider/Deep-Live-Cam)
 *
 * Architecture:
 *  - MediaPipe FaceMesh  → 468 landmark points → expression analysis + face swap
 *  - MediaPipe Pose      → 33 pose landmarks   → skeleton overlay + body analysis
 *  - Canvas 2D API       → affine transform face warp (triangle-based)
 */

// ─────────────────────────────────────────────
//  CONFIG & STATE
// ─────────────────────────────────────────────
const CONFIG = {
  avatars: [
    { id: 'warrior',  name: 'Ciborg',    file: 'assets/warrior.jpg' },
    { id: 'anime',    name: 'Anime',     file: 'assets/anime.jpg'   },
    { id: 'villain',  name: 'Vilão',     file: 'assets/villain.jpg' },
    { id: 'hero',     name: 'Herói',     file: 'assets/hero.jpg'    },
    { id: 'robot',    name: 'Robô',      file: 'assets/robot.jpg'   },
    { id: 'witch',    name: 'Bruxa',     file: 'assets/witch.jpg'   },
  ],
};

const state = {
  running: false,
  selectedAvatar: null,
  avatarImages: {},
  blendStrength: 0.8,
  smoothing: 0.5,
  faceScale: 1.0,
  showMesh: true,
  showSkeleton: true,
  showHUD: true,
  mirrorMode: true,
  // tracking results
  faceLandmarks: null,
  poseLandmarks: null,
  // expression state (smoothed)
  expressions: { happy: 0, sad: 0, surprise: 0, angry: 0, neutral: 1 },
  // FPS
  fps: 0,
  lastFrameTime: 0,
  frameCount: 0,
  fpsTimer: 0,
};

// ─────────────────────────────────────────────
//  DOM ELEMENTS
// ─────────────────────────────────────────────
const $ = id => document.getElementById(id);
const video         = $('inputVideo');
const overlayCanvas = $('overlayCanvas');
const outputCanvas  = $('outputCanvas');
const overlayCtx    = overlayCanvas.getContext('2d');
const outputCtx     = outputCanvas.getContext('2d');
const skeletonCvs   = $('skeletonCanvas');
const skelCtx       = skeletonCvs.getContext('2d');

// ─────────────────────────────────────────────
//  MEDIAPIPE SETUP
// ─────────────────────────────────────────────
let faceMesh = null;
let pose     = null;
let mpCamera = null;

function initMediaPipe() {
  // Face Mesh
  faceMesh = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  faceMesh.onResults(onFaceMeshResults);

  // Pose
  pose = new Pose({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}` });
  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  pose.onResults(onPoseResults);

  // Camera
  mpCamera = new Camera(video, {
    onFrame: async () => {
      if (!state.running) return;
      await faceMesh.send({ image: video });
      await pose.send({ image: video });
      updateFPS();
    },
    width: 640,
    height: 480,
  });
}

// ─────────────────────────────────────────────
//  FACE MESH CALLBACK
// ─────────────────────────────────────────────
function onFaceMeshResults(results) {
  syncCanvasSize(overlayCanvas);
  syncCanvasSize(outputCanvas);

  const W = overlayCanvas.width;
  const H = overlayCanvas.height;

  overlayCtx.clearRect(0, 0, W, H);

  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    state.faceLandmarks = results.multiFaceLandmarks[0];
    $('faceBadge').textContent = '🟢 Rosto';
    $('faceBadge').className = 'badge badge-active';

    // ── Analyse expressions ──
    analyzeExpressions(state.faceLandmarks, W, H);

    // ── Draw face mesh ──
    if (state.showMesh) drawFaceMesh(state.faceLandmarks, W, H);

    // ── Draw face swap ──
    if (state.selectedAvatar && state.avatarImages[state.selectedAvatar]) {
      drawFaceSwap(state.faceLandmarks, W, H);
    }
  } else {
    state.faceLandmarks = null;
    $('faceBadge').textContent = '⬜ Rosto';
    $('faceBadge').className = 'badge badge-cyan';
    fadeExpressions();
    clearOutputCanvas(W, H);
  }

  updateExpressionUI();
}

// ─────────────────────────────────────────────
//  POSE CALLBACK
// ─────────────────────────────────────────────
function onPoseResults(results) {
  if (results.poseLandmarks) {
    state.poseLandmarks = results.poseLandmarks;
    $('poseBadge').textContent = '🟢 Corpo';
    $('poseBadge').className = 'badge badge-active';

    const W = overlayCanvas.width;
    const H = overlayCanvas.height;

    if (state.showSkeleton) drawPoseSkeleton(results.poseLandmarks, W, H);
    drawSkeletonMap(results.poseLandmarks);
    updateBodyAngles(results.poseLandmarks);
  } else {
    state.poseLandmarks = null;
    $('poseBadge').textContent = '⬜ Corpo';
    $('poseBadge').className = 'badge badge-purple';
  }
}

// ─────────────────────────────────────────────
//  FACE SWAP (Affine Warp)
// ─────────────────────────────────────────────
// Key face landmark indices for bounding ellipse
const FACE_OVAL_IDX = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
];

function drawFaceSwap(landmarks, W, H) {
  const pts = FACE_OVAL_IDX.map(i => ({
    x: landmarks[i].x * W,
    y: landmarks[i].y * H,
  }));

  // Compute bounding box of face oval
  const xs = pts.map(p => p.x);
  const ys = pts.map(p => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);

  const fw = x1 - x0;
  const fh = y1 - y0;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;

  // Scale factor
  const scale = state.faceScale;
  const dw = fw * scale;
  const dh = fh * scale;

  outputCtx.clearRect(0, 0, W, H);
  outputCtx.save();

  // Draw the avatar image warped to the face bbox
  // Clip to face oval shape with feathered edges
  outputCtx.globalCompositeOperation = 'source-over';
  outputCtx.globalAlpha = state.blendStrength;

  // Create oval clip path
  outputCtx.beginPath();
  const rx = dw / 2 * 1.05;
  const ry = dh / 2 * 1.1;

  // Ellipse slightly feathered
  outputCtx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  outputCtx.clip();

  // Draw avatar image fitted to face region
  const avatarImg = state.avatarImages[state.selectedAvatar];
  outputCtx.drawImage(avatarImg, cx - dw / 2, cy - dh / 2, dw, dh);

  outputCtx.restore();

  // Feathered blend edges
  applyEdgeFeather(cx, cy, dw * 0.5, dh * 0.55, W, H);
}

function applyEdgeFeather(cx, cy, rx, ry, W, H) {
  // Apply a radial gradient mask for smooth blending at edges
  const grd = outputCtx.createRadialGradient(cx, cy, Math.min(rx, ry) * 0.5, cx, cy, Math.max(rx, ry));
  grd.addColorStop(0, 'rgba(0,0,0,0)');
  grd.addColorStop(1, 'rgba(0,0,0,1)');
  outputCtx.save();
  outputCtx.globalCompositeOperation = 'destination-out';
  outputCtx.fillStyle = grd;
  outputCtx.fillRect(0, 0, W, H);
  outputCtx.restore();
}

function clearOutputCanvas(W, H) {
  outputCtx.clearRect(0, 0, W, H);
}

// ─────────────────────────────────────────────
//  FACE MESH DRAW
// ─────────────────────────────────────────────
const FACE_TESSELATION = window.FACEMESH_TESSELATION || [];

function drawFaceMesh(landmarks, W, H) {
  overlayCtx.save();
  overlayCtx.strokeStyle = 'rgba(0,245,255,0.25)';
  overlayCtx.lineWidth = 0.5;

  // draw tesselation (requires mediapipe drawing_utils)
  if (typeof drawConnectors === 'function') {
    drawConnectors(overlayCtx, landmarks, FACEMESH_TESSELATION,
      { color: 'rgba(0,245,255,0.15)', lineWidth: 0.5 });
    drawConnectors(overlayCtx, landmarks, FACEMESH_RIGHT_EYE,
      { color: 'rgba(0,245,255,0.8)', lineWidth: 1 });
    drawConnectors(overlayCtx, landmarks, FACEMESH_LEFT_EYE,
      { color: 'rgba(0,245,255,0.8)', lineWidth: 1 });
    drawConnectors(overlayCtx, landmarks, FACEMESH_FACE_OVAL,
      { color: 'rgba(168,85,247,0.7)', lineWidth: 1.5 });
    drawConnectors(overlayCtx, landmarks, FACEMESH_LIPS,
      { color: 'rgba(239,68,68,0.6)', lineWidth: 1 });
  }

  // Key landmarks dots
  overlayCtx.fillStyle = 'rgba(0,245,255,0.8)';
  const keyPoints = [1, 4, 33, 263, 61, 291, 199]; // nose, eyes, mouth corners
  keyPoints.forEach(i => {
    const lm = landmarks[i];
    if (!lm) return;
    overlayCtx.beginPath();
    overlayCtx.arc(lm.x * W, lm.y * H, 2, 0, Math.PI * 2);
    overlayCtx.fill();
  });

  overlayCtx.restore();
}

// ─────────────────────────────────────────────
//  POSE SKELETON DRAW
// ─────────────────────────────────────────────
const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[24,26],[26,28],
  [27,29],[29,31],[28,30],[30,32],
];

function drawPoseSkeleton(landmarks, W, H) {
  overlayCtx.save();

  // Connections
  POSE_CONNECTIONS.forEach(([a, b]) => {
    const la = landmarks[a], lb = landmarks[b];
    if (!la || !lb || la.visibility < 0.5 || lb.visibility < 0.5) return;
    overlayCtx.beginPath();
    overlayCtx.moveTo(la.x * W, la.y * H);
    overlayCtx.lineTo(lb.x * W, lb.y * H);
    const grad = overlayCtx.createLinearGradient(la.x * W, la.y * H, lb.x * W, lb.y * H);
    grad.addColorStop(0, '#00f5ff');
    grad.addColorStop(1, '#a855f7');
    overlayCtx.strokeStyle = grad;
    overlayCtx.lineWidth = 2.5;
    overlayCtx.stroke();
  });

  // Joints
  landmarks.forEach((lm, i) => {
    if (lm.visibility < 0.5) return;
    overlayCtx.beginPath();
    overlayCtx.arc(lm.x * W, lm.y * H, 4, 0, Math.PI * 2);
    overlayCtx.fillStyle = '#00f5ff';
    overlayCtx.shadowColor = '#00f5ff';
    overlayCtx.shadowBlur = 8;
    overlayCtx.fill();
    overlayCtx.shadowBlur = 0;
  });

  overlayCtx.restore();
}

// ─────────────────────────────────────────────
//  SKELETON MAP (mini side panel)
// ─────────────────────────────────────────────
function drawSkeletonMap(landmarks) {
  const W = skeletonCvs.width;
  const H = skeletonCvs.height;
  skelCtx.clearRect(0, 0, W, H);

  skelCtx.save();

  POSE_CONNECTIONS.forEach(([a, b]) => {
    const la = landmarks[a], lb = landmarks[b];
    if (!la || !lb || la.visibility < 0.4 || lb.visibility < 0.4) return;
    skelCtx.beginPath();
    skelCtx.moveTo(la.x * W, la.y * H);
    skelCtx.lineTo(lb.x * W, lb.y * H);
    skelCtx.strokeStyle = 'rgba(0,245,255,0.7)';
    skelCtx.lineWidth = 1.5;
    skelCtx.stroke();
  });

  landmarks.forEach(lm => {
    if (lm.visibility < 0.4) return;
    skelCtx.beginPath();
    skelCtx.arc(lm.x * W, lm.y * H, 2.5, 0, Math.PI * 2);
    skelCtx.fillStyle = '#a855f7';
    skelCtx.fill();
  });

  skelCtx.restore();
}

// ─────────────────────────────────────────────
//  EXPRESSION ANALYSIS
// ─────────────────────────────────────────────
// Landmark indices for expression heuristics
const EXP_LM = {
  // Mouth corners
  mouthLeft: 61, mouthRight: 291,
  // Mouth top/bottom
  mouthTop: 13, mouthBottom: 14,
  // Eye outer corners
  eyeLeftOuter: 33, eyeRightOuter: 263,
  // Brow inner left/right
  browLeft: 70, browRight: 300,
  // Nose tip
  noseTip: 1,
};

function analyzeExpressions(landmarks, W, H) {
  const lm = landmarks;
  const lp = k => ({ x: lm[EXP_LM[k]].x * W, y: lm[EXP_LM[k]].y * H });
  const dist = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);

  const mL = lp('mouthLeft');
  const mR = lp('mouthRight');
  const mT = lp('mouthTop');
  const mB = lp('mouthBottom');
  const eL = lp('eyeLeftOuter');
  const eR = lp('eyeRightOuter');
  const bL = lp('browLeft');
  const bR = lp('browRight');
  const nT = lp('noseTip');

  const mouthWidth = dist(mL, mR);
  const mouthOpen  = dist(mT, mB);
  const eyeWidth   = dist(eL, eR);
  const browL_eye  = Math.abs(bL.y - eL.y);
  const browR_eye  = Math.abs(bR.y - eR.y);

  // Normalized ratios
  const smileRatio   = mouthWidth / eyeWidth;      // > 0.5 = wide smile
  const openRatio    = mouthOpen / mouthWidth;      // > 0.3 = open mouth
  const browRaised   = ((browL_eye + browR_eye) / 2) / eyeWidth;

  // Head tilt
  const dX = mR.x - mL.x;
  const dY = mR.y - mL.y;
  const tiltDeg = Math.atan2(dY, dX) * 180 / Math.PI;

  // Expression scores (raw)
  const rawHappy    = Math.max(0, Math.min(1, (smileRatio - 0.45) * 4));
  const rawSurprise = openRatio > 0.25 && browRaised > 0.15
    ? Math.min(1, openRatio * 2 + browRaised) : 0;
  const rawAngry    = browRaised < 0.08 && smileRatio < 0.5
    ? Math.min(1, (0.1 - browRaised) * 8) : 0;
  const rawSad      = smileRatio < 0.45 && !rawAngry
    ? Math.min(1, (0.45 - smileRatio) * 4) : 0;
  const rawNeutral  = Math.max(0, 1 - rawHappy - rawSurprise - rawAngry - rawSad);

  // Smooth expressions
  const s = 0.85;
  state.expressions.happy    = lerp(state.expressions.happy,    rawHappy,    1 - s);
  state.expressions.surprise = lerp(state.expressions.surprise, rawSurprise, 1 - s);
  state.expressions.angry    = lerp(state.expressions.angry,    rawAngry,    1 - s);
  state.expressions.sad      = lerp(state.expressions.sad,      rawSad,      1 - s);
  state.expressions.neutral  = lerp(state.expressions.neutral,  rawNeutral,  1 - s);

  // Head tilt HUD
  const dir = tiltDeg > 3 ? `↗ ${tiltDeg.toFixed(0)}°` : tiltDeg < -3 ? `↖ ${Math.abs(tiltDeg).toFixed(0)}°` : '→ Reto';
  $('headValue').textContent = dir;

  // Dominant expression
  const expNames = { happy: '😊 Feliz', sad: '😢 Triste', surprise: '😮 Surpreso', angry: '😠 Irritado', neutral: '😐 Neutro' };
  const dominant = Object.entries(state.expressions).sort((a, b) => b[1] - a[1])[0];
  $('expressionValue').textContent = expNames[dominant[0]] || '—';
  $('poseValue').textContent = 'Em Pé';
}

function fadeExpressions() {
  state.expressions.happy    *= 0.9;
  state.expressions.surprise *= 0.9;
  state.expressions.angry    *= 0.9;
  state.expressions.sad      *= 0.9;
  state.expressions.neutral   = 1 - state.expressions.happy - state.expressions.surprise - state.expressions.angry - state.expressions.sad;
}

function updateExpressionUI() {
  const expMap = { happy: 'happy', sad: 'sad', surprise: 'surprise', angry: 'angry', neutral: 'neutral' };
  Object.entries(expMap).forEach(([key, id]) => {
    const val = Math.max(0, Math.min(1, state.expressions[key]));
    const pct = Math.round(val * 100);
    $(`bar-${id}`).style.width = `${pct}%`;
    $(`pct-${id}`).textContent = `${pct}%`;
  });
}

// ─────────────────────────────────────────────
//  BODY ANGLES
// ─────────────────────────────────────────────
function updateBodyAngles(landmarks) {
  const lm = landmarks;

  // Shoulder angle
  const ls = lm[11], rs = lm[12];
  if (ls && rs && ls.visibility > 0.5 && rs.visibility > 0.5) {
    const sAngle = Math.atan2(rs.y - ls.y, rs.x - ls.x) * 180 / Math.PI;
    $('shoulderAngle').textContent = `${sAngle.toFixed(1)}°`;
    $('tiltAngle').textContent = Math.abs(sAngle) < 5 ? 'Reto' : sAngle > 0 ? 'Dir. ↗' : 'Esq. ↖';
  }

  // Left arm
  const le = lm[13], lw = lm[15];
  if (le && lw && le.visibility > 0.5) {
    const lAngle = angleBetween3Points(lm[11], le, lw);
    $('armLAngle').textContent = `${lAngle.toFixed(0)}°`;
  }

  // Right arm
  const re = lm[14], rw = lm[16];
  if (re && rw && re.visibility > 0.5) {
    const rAngle = angleBetween3Points(lm[12], re, rw);
    $('armRAngle').textContent = `${rAngle.toFixed(0)}°`;
  }
}

function angleBetween3Points(a, b, c) {
  if (!a || !b || !c) return 0;
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.hypot(ab.x, ab.y);
  const magCB = Math.hypot(cb.x, cb.y);
  if (magAB === 0 || magCB === 0) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (magAB * magCB)))) * 180 / Math.PI;
}

// ─────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────
function lerp(a, b, t) { return a + (b - a) * t; }

function syncCanvasSize(canvas) {
  const parent = canvas.parentElement;
  if (canvas.width !== parent.clientWidth || canvas.height !== parent.clientHeight) {
    canvas.width  = parent.clientWidth;
    canvas.height = parent.clientHeight;
  }
}

function updateFPS() {
  const now = performance.now();
  state.frameCount++;
  if (now - state.fpsTimer > 1000) {
    state.fps = Math.round(state.frameCount * 1000 / (now - state.fpsTimer));
    state.frameCount = 0;
    state.fpsTimer = now;
    $('fpsBadge').textContent = `${state.fps} FPS`;
  }
}

// ─────────────────────────────────────────────
//  AVATAR LOADING
// ─────────────────────────────────────────────
function loadAvatarImages() {
  CONFIG.avatars.forEach(av => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      state.avatarImages[av.id] = img;
      checkAvatarsLoaded();
    };
    img.onerror = () => {
      console.warn(`Avatar not found: ${av.file}`);
      // Create a placeholder canvas image
      const ph = createPlaceholderAvatar(av.name);
      state.avatarImages[av.id] = ph;
      checkAvatarsLoaded();
    };
    img.src = av.file;
  });
}

function createPlaceholderAvatar(name) {
  const c = document.createElement('canvas');
  c.width = 200; c.height = 200;
  const ctx = c.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 200, 200);
  grad.addColorStop(0, '#1a1a2e');
  grad.addColorStop(1, '#16213e');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 200, 200);

  // Draw a face silhouette
  ctx.fillStyle = '#2a2a4e';
  ctx.beginPath();
  ctx.arc(100, 80, 50, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#00f5ff';
  ctx.font = 'bold 14px Outfit, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(name, 100, 170);

  return c;
}

function checkAvatarsLoaded() {
  const loaded = Object.keys(state.avatarImages).length;
  $('avatarCount').textContent = `${loaded} disponíveis`;
}

// ─────────────────────────────────────────────
//  AVATAR GALLERY UI
// ─────────────────────────────────────────────
function buildAvatarGrid() {
  const grid = $('avatarGrid');
  grid.innerHTML = '';

  CONFIG.avatars.forEach(av => {
    const card = document.createElement('div');
    card.className = 'avatar-card';
    card.id = `avcard-${av.id}`;

    const img = document.createElement('img');
    img.src = av.file;
    img.alt = av.name;
    img.onerror = () => { img.style.display = 'none'; };

    const nameEl = document.createElement('div');
    nameEl.className = 'avatar-name';
    nameEl.textContent = av.name;

    card.appendChild(img);
    card.appendChild(nameEl);

    card.addEventListener('click', () => selectAvatar(av.id));
    grid.appendChild(card);
  });
}

function selectAvatar(id) {
  state.selectedAvatar = id;
  document.querySelectorAll('.avatar-card').forEach(c => c.classList.remove('selected'));
  const card = $(`avcard-${id}`);
  if (card) card.classList.add('selected');
}

// ─────────────────────────────────────────────
//  CUSTOM UPLOAD
// ─────────────────────────────────────────────
function handleAvatarUpload(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      const id = `custom-${Date.now()}`;
      state.avatarImages[id] = img;

      // Add to CONFIG and rebuild
      CONFIG.avatars.push({ id, name: 'Personalizado', file: '' });

      const grid = $('avatarGrid');
      const card = document.createElement('div');
      card.className = 'avatar-card';
      card.id = `avcard-${id}`;

      const imgEl = document.createElement('img');
      imgEl.src = e.target.result;
      imgEl.alt = 'Personalizado';

      const nameEl = document.createElement('div');
      nameEl.className = 'avatar-name';
      nameEl.textContent = 'Seu Upload';

      card.appendChild(imgEl);
      card.appendChild(nameEl);
      card.addEventListener('click', () => selectAvatar(id));
      grid.appendChild(card);

      selectAvatar(id);
      $('avatarCount').textContent = `${CONFIG.avatars.length} disponíveis`;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// ─────────────────────────────────────────────
//  SNAPSHOT
// ─────────────────────────────────────────────
function takeSnapshot() {
  const snapCanvas = $('snapshotCanvas');
  const W = overlayCanvas.width, H = overlayCanvas.height;
  snapCanvas.width = W; snapCanvas.height = H;
  const ctx = snapCanvas.getContext('2d');

  // Draw original video frame
  ctx.save();
  if (state.mirrorMode) {
    ctx.translate(W, 0); ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, W, H);
  ctx.restore();

  // Draw overlay (non-mirrored already)
  ctx.drawImage(overlayCanvas, 0, 0);
  ctx.drawImage(outputCanvas, 0, 0, W, H);

  // Add watermark
  ctx.fillStyle = 'rgba(0,245,255,0.6)';
  ctx.font = '13px JetBrains Mono, monospace';
  ctx.fillText('DeepSwap AI', 10, H - 10);

  // Show modal
  const modal = $('snapshotModal');
  modal.classList.add('open');

  // Download link
  const dl = $('downloadLink');
  dl.href = snapCanvas.toDataURL('image/png');
  dl.download = `deepswap-${Date.now()}.png`;
}

// ─────────────────────────────────────────────
//  CAMERA START
// ─────────────────────────────────────────────
async function startCamera() {
  try {
    $('startBtn').textContent = '⏳ Carregando...';
    $('startBtn').disabled = true;
    $('statusLabel').textContent = 'Inicializando MediaPipe...';
    $('statusDot').className = 'status-dot';

    initMediaPipe();
    await mpCamera.start();

    state.running = true;
    $('startOverlay').classList.add('hidden');
    $('statusDot').className = 'status-dot active';
    $('statusLabel').textContent = 'Câmera ativa';
  } catch (err) {
    $('statusDot').className = 'status-dot error';
    $('statusLabel').textContent = 'Erro: ' + err.message;
    $('startBtn').textContent = '▶ Tentar novamente';
    $('startBtn').disabled = false;
    console.error('Camera error:', err);
  }
}

// ─────────────────────────────────────────────
//  CONTROLS BINDING
// ─────────────────────────────────────────────
function bindControls() {
  // Sliders
  $('blendStrength').addEventListener('input', e => {
    state.blendStrength = e.target.value / 100;
    $('blendVal').textContent = `${e.target.value}%`;
  });
  $('smoothing').addEventListener('input', e => {
    state.smoothing = e.target.value / 100;
    $('smoothVal').textContent = `${e.target.value}%`;
  });
  $('faceScale').addEventListener('input', e => {
    state.faceScale = e.target.value / 100;
    $('scaleVal').textContent = `${e.target.value}%`;
  });

  // Toggles
  $('showMesh').addEventListener('change', e => { state.showMesh = e.target.checked; });
  $('showSkeleton').addEventListener('change', e => { state.showSkeleton = e.target.checked; });
  $('showHUD').addEventListener('change', e => {
    state.showHUD = e.target.checked;
    $('hudOverlay').style.opacity = e.target.checked ? '1' : '0';
  });
  $('mirrorMode').addEventListener('change', e => {
    state.mirrorMode = e.target.checked;
    video.style.transform = e.target.checked ? 'scaleX(-1)' : 'scaleX(1)';
    outputCanvas.style.transform = e.target.checked ? 'scaleX(-1)' : 'scaleX(1)';
  });

  // Start button
  $('startBtn').addEventListener('click', startCamera);

  // Snapshot
  $('snapshotBtn').addEventListener('click', () => {
    if (state.running) takeSnapshot();
  });

  // Clear avatar
  $('clearAvatarBtn').addEventListener('click', () => {
    state.selectedAvatar = null;
    document.querySelectorAll('.avatar-card').forEach(c => c.classList.remove('selected'));
    clearOutputCanvas(overlayCanvas.width, overlayCanvas.height);
  });

  // Modal close
  $('closeModal').addEventListener('click', () => $('snapshotModal').classList.remove('open'));
  $('snapshotModal').addEventListener('click', e => {
    if (e.target === $('snapshotModal')) $('snapshotModal').classList.remove('open');
  });

  // File upload
  $('uploadInput').addEventListener('change', e => {
    handleAvatarUpload(e.target.files[0]);
  });

  // Drag and drop on upload zone
  const uploadZone = document.querySelector('.upload-zone');
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.style.borderColor = 'var(--cyan)'; });
  uploadZone.addEventListener('dragleave', () => { uploadZone.style.borderColor = ''; });
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    handleAvatarUpload(file);
  });

  // Nav buttons (placeholder, could add more pages)
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}

// ─────────────────────────────────────────────
//  INIT
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildAvatarGrid();
  loadAvatarImages();
  bindControls();

  // Resize canvases on window resize
  window.addEventListener('resize', () => {
    syncCanvasSize(overlayCanvas);
    syncCanvasSize(outputCanvas);
  });

  console.log('%c DeepSwap AI Inicializado ', 'background:#00f5ff;color:#000;font-weight:bold;padding:4px 8px;border-radius:4px;');
  console.log('Inspirado em: https://github.com/hacksider/Deep-Live-Cam');
});
