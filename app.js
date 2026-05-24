// ============================================================
//  DAVE-2 EDGE AI — app.js
//  Self-Driving Car Demo · TensorFlow.js + Voice Control
//  IU Project: DLBAIPEAI – Project Edge AI
// ============================================================

// ── State ────────────────────────────────────────────────────
let model        = null;
let isRunning    = false;
let voiceActive  = false;
let manualMode   = false;
let manualAngle  = 0;
let recognition  = null;
let animId       = null;

let interventions       = 0;
let interventionActive  = false;
let startTime           = null;
let steeringSmooth      = 0;
let lastFrameTime   = performance.now();
let frameCount      = 0;
let fpsDisplay      = 0;

// ── DOM refs ──────────────────────────────────────────────────
const video        = document.getElementById('camera');
const canvas       = document.getElementById('canvas');
const ctx          = canvas.getContext('2d');
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');
const modelBadge   = document.getElementById('model-badge');
const gpuBadge     = document.getElementById('gpu-badge');
const startBtn     = document.getElementById('start-btn');
const voiceBtn     = document.getElementById('voice-btn');
const leftBtn      = document.getElementById('left-btn');
const rightBtn     = document.getElementById('right-btn');
const angleVal     = document.getElementById('angle-value');
const directionVal = document.getElementById('direction-val');
const dirArrow     = document.getElementById('direction-arrow');
const needle       = document.getElementById('steering-needle');
const autonomyVal  = document.getElementById('autonomy-val');
const autonomyFill = document.getElementById('autonomy-fill');
const intVal       = document.getElementById('intervention-val');
const elapsedVal   = document.getElementById('elapsed-val');
const voiceInd     = document.getElementById('voice-indicator');
const voiceStatus  = document.getElementById('voice-status-text');
const voiceHeard   = document.getElementById('voice-heard');
const fpsDom       = document.getElementById('fps-display');
const pathPoly     = document.getElementById('path-poly');
const centerLine   = document.getElementById('center-line');

// ── Initialise ────────────────────────────────────────────────
window.addEventListener('load', async () => {
  setStatus('warn', 'Starting camera...');
  await startCamera();
  setStatus('warn', 'Loading DAVE-2 model...');
  await loadModel();
  bindManualButtons();
});

// ── Camera ────────────────────────────────────────────────────
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(r => video.onloadedmetadata = r);
    setStatus('ok', 'Camera ready');
  } catch (e) {
    setStatus('err', 'Camera error: ' + e.message);
  }
}

// ── Model ─────────────────────────────────────────────────────
async function loadModel() {
  try {
    model = await tf.loadGraphModel('./tfjs_graph/model.json');

    // Warm-up inference
    const dummy = tf.zeros([1, 66, 200, 3]);
    model.predict(dummy).dispose();
    dummy.dispose();

    // Check GPU
    const backend = tf.getBackend();
    if (backend === 'webgl') {
      gpuBadge.textContent = 'WebGL GPU';
      gpuBadge.classList.add('active');
    } else {
      gpuBadge.textContent = 'CPU';
    }

    modelBadge.textContent = 'READY';
    modelBadge.classList.add('active');
    setStatus('ok', 'DAVE-2 model loaded · Ready to drive');

  } catch (e) {
    modelBadge.textContent = 'ERROR';
    setStatus('err', 'Model load failed: ' + e.message);
    console.error(e);
  }
}

// ── Preprocess ────────────────────────────────────────────────
// NOTE: Normalization handled by Rescaling layer inside model
// We pass raw pixel values (0-255) — model converts to -1..+1
function preprocessFrame() {
  try {
    ctx.drawImage(video, 0, 0, 200, 66);
    return tf.tidy(() => {
      const t = tf.browser.fromPixels(canvas)
                  .toFloat()   // 0-255 raw → Rescaling layer handles the rest
                  .expandDims(0);
      return t;
    });
  } catch { return null; }
}

// ── Drive Loop ────────────────────────────────────────────────
function toggleDriving() {
  if (!model) { setStatus('err', 'Model not loaded yet'); return; }
  isRunning = !isRunning;

  if (isRunning) {
    startTime = startTime || Date.now();
    startBtn.textContent  = '⏸ PAUSE';
    startBtn.classList.add('running');
    setStatus('ok', 'Autonomous driving active');
    driveLoop();
  } else {
    startBtn.innerHTML = '<span class="btn-icon">▶</span> START';
    startBtn.classList.remove('running');
    cancelAnimationFrame(animId);
    setStatus('warn', 'Paused');
  }
}

async function driveLoop() {
  if (!isRunning) return;

  // FPS counter
  frameCount++;
  const now = performance.now();
  if (now - lastFrameTime >= 1000) {
    fpsDisplay = frameCount;
    frameCount = 0;
    lastFrameTime = now;
    fpsDom.textContent = fpsDisplay + ' FPS';
  }

  let angle = 0;

  if (manualMode) {
    // Human override — count only ONCE per activation
    angle = manualAngle;
    if (!interventionActive) {
      interventions++;
      intVal.textContent = interventions;
      interventionActive = true;
    }
    document.body.classList.add('manual-active');
  } else {
    interventionActive = false;
    document.body.classList.remove('manual-active');
    // AI prediction
    const tensor = preprocessFrame();
    if (tensor) {
      const pred = model.predict(tensor);
      angle = (await pred.data())[0];
      pred.dispose();
      tensor.dispose();
    }
  }

  // Smooth steering (exponential moving average)
  steeringSmooth = 0.82 * steeringSmooth + 0.18 * angle;

  updateHUD(steeringSmooth);
  updateAutonomy();
  updatePathVisualization(steeringSmooth);

  animId = requestAnimationFrame(driveLoop);
}

// ── HUD Update ────────────────────────────────────────────────
function updateHUD(angle) {
  // Angle display
  angleVal.textContent = angle.toFixed(3);

  // Direction
  let dir = 'STRAIGHT';
  let arrow = '↑';
  if (angle < -0.08) { dir = 'LEFT';  arrow = '↖'; }
  if (angle > 0.08)  { dir = 'RIGHT'; arrow = '↗'; }

  directionVal.textContent = dir;
  dirArrow.textContent     = arrow;

  // Steering needle — map -1..1 to 5%..95% of track width
  const pct = ((angle + 1) / 2) * 90 + 5;
  needle.style.left = Math.max(5, Math.min(95, pct)) + '%';

  // Color the needle red if extreme
  if (Math.abs(angle) > 0.5) {
    needle.style.background = 'var(--danger)';
    needle.style.boxShadow  = '0 0 6px var(--danger)';
  } else {
    needle.style.background = 'var(--accent)';
    needle.style.boxShadow  = '0 0 6px var(--accent)';
  }
}

function updateAutonomy() {
  if (!startTime) return;
  const elapsed = Math.max(1, (Date.now() - startTime) / 1000);
  const grade   = Math.max(0, (1 - (interventions * 6) / elapsed) * 100);

  autonomyVal.textContent  = grade.toFixed(1) + '%';
  autonomyFill.style.width = grade + '%';
  elapsedVal.textContent   = Math.round(elapsed) + 's elapsed';

  // Colour by grade
  if (grade >= 80) {
    autonomyFill.style.background = 'var(--accent)';
  } else if (grade >= 50) {
    autonomyFill.style.background = 'var(--warning)';
  } else {
    autonomyFill.style.background = 'var(--danger)';
  }
}

// ── Path Visualization ────────────────────────────────────────
function updatePathVisualization(angle) {
  // Offset the predicted path polygon based on steering angle
  const offset = angle * 18; // scale to SVG coords
  const cx = 50 + offset;

  pathPoly.setAttribute('points',
    `${38},100 ${62},100 ${cx + 6},58 ${cx - 6},58`
  );
  centerLine.setAttribute('x1', '50');
  centerLine.setAttribute('y1', '100');
  centerLine.setAttribute('x2', String(cx));
  centerLine.setAttribute('y2', '58');
}

// ── Emergency Stop ────────────────────────────────────────────
function emergencyStop() {
  isRunning          = false;
  manualMode         = false;
  manualAngle        = 0;
  steeringSmooth     = 0;
  interventionActive = false;
  cancelAnimationFrame(animId);
  document.body.classList.remove('manual-active');
  startBtn.innerHTML = '<span class="btn-icon">▶</span> START';
  startBtn.classList.remove('running');
  setStatus('err', 'EMERGENCY STOP activated');
  updateHUD(0);
}

// ── Manual Buttons ────────────────────────────────────────────
function bindManualButtons() {
  const activate = (dir) => {
    manualMode  = true;
    manualAngle = dir === 'left' ? -0.45 : 0.45;
  };
  const deactivate = () => {
    manualMode  = false;
    manualAngle = 0;
    document.body.classList.remove('manual-active');
  };

  ['pointerdown', 'touchstart'].forEach(e => {
    leftBtn.addEventListener(e,  () => activate('left'),  { passive: true });
    rightBtn.addEventListener(e, () => activate('right'), { passive: true });
  });
  ['pointerup', 'touchend', 'pointerleave'].forEach(e => {
    leftBtn.addEventListener(e,  deactivate, { passive: true });
    rightBtn.addEventListener(e, deactivate, { passive: true });
  });
}

// ── Voice Control ─────────────────────────────────────────────
function toggleVoice() {
  voiceActive = !voiceActive;

  if (voiceActive) {
    startVoice();
    voiceBtn.classList.add('active');
    voiceBtn.innerHTML = '<span class="btn-icon">🔴</span> LISTENING';
    voiceStatus.textContent = 'Voice ON';
    voiceStatus.classList.add('active');
    voiceInd.classList.add('listening');
  } else {
    if (recognition) recognition.stop();
    voiceBtn.classList.remove('active');
    voiceBtn.innerHTML = '<span class="btn-icon">🎤</span> VOICE';
    voiceStatus.textContent = 'Voice OFF';
    voiceStatus.classList.remove('active');
    voiceInd.classList.remove('listening');
  }
}

function startVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    alert('Speech Recognition not supported. Please use Safari on iPhone.');
    voiceActive = false;
    return;
  }

  recognition = new SR();
  recognition.continuous     = true;
  recognition.interimResults = false;
  recognition.lang           = 'en-US';

  recognition.onresult = (e) => {
    const cmd = e.results[e.results.length - 1][0].transcript
                  .trim().toLowerCase();
    voiceHeard.textContent = '"' + cmd + '"';
    handleCommand(cmd);
  };

  recognition.onerror = (e) => {
    voiceHeard.textContent = 'Error: ' + e.error;
  };

  recognition.onend = () => {
    if (voiceActive) recognition.start(); // auto-restart
  };

  recognition.start();
}

function handleCommand(cmd) {
  if (cmd.includes('left')) {
    triggerManual(-0.5, 1200);
    flashStatus('◀ VOICE: LEFT', 'warn');

  } else if (cmd.includes('right')) {
    triggerManual(0.5, 1200);
    flashStatus('▶ VOICE: RIGHT', 'warn');

  } else if (cmd.includes('stop') || cmd.includes('brake') || cmd.includes('halt')) {
    emergencyStop();
    flashStatus('⛔ VOICE: STOP', 'err');

  } else if (cmd.includes('go') || cmd.includes('start') || cmd.includes('drive')) {
    manualMode = false;
    if (!isRunning) toggleDriving();
    flashStatus('▶ VOICE: GO', 'ok');

  } else if (cmd.includes('straight') || cmd.includes('forward')) {
    triggerManual(0, 800);
    flashStatus('↑ VOICE: STRAIGHT', 'ok');
  }
}

function triggerManual(angle, duration) {
  manualMode  = true;
  manualAngle = angle;
  setTimeout(() => {
    manualMode  = false;
    manualAngle = 0;
  }, duration);
}

// ── Helpers ───────────────────────────────────────────────────
function setStatus(type, msg) {
  statusDot.className = 'status-dot ' + type;
  statusText.textContent = msg.toUpperCase();
}

let flashTimer = null;
function flashStatus(msg, type) {
  setStatus(type, msg);
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => {
    setStatus('ok', 'Autonomous driving active');
  }, 2000);
}