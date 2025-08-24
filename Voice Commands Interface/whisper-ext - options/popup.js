// popup.js
const listenBtn = document.getElementById('listenBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const transcriptEl = document.getElementById('transcript');

let audioStream = null;
let mediaRecorder = null;
let chunks = [];
let audioCtx = null;
let analyser = null;
let dataArray = null;
let monitoringTimer = null;

const START_RMS = 0.01;
const SILENCE_RMS = 0.008;
const START_PERSIST_MS = 120;
const SILENCE_TIMEOUT_MS = 2000;

let startDetectedSince = null;
let silenceStartedAt = null;
let isRecording = false;

function setStatus(s) { statusEl.textContent = 'Status: ' + s; }

// options (loaded from chrome.storage.sync)
let extOptions = {
  saveLocal: true,
  filenamePattern: 'whisper/recording-{date}-{time}.webm',
  alsoUpload: true
};

function loadOptionsIntoMemory() {
  chrome.storage.sync.get(extOptions, (items) => {
    extOptions = Object.assign(extOptions, items);
    console.log('Loaded extension options:', extOptions);
  });
}
loadOptionsIntoMemory();
// keep listening for changes (optional)
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'sync') {
    for (const k in changes) extOptions[k] = changes[k].newValue;
    console.log('Options changed:', extOptions);
  }
});


/**
 * Open a small permission window (recorder.html) and wait until microphone permission becomes granted.
 * Returns true if permission granted within timeoutMs, false otherwise.
 */
async function openPermissionWindowAndWait(timeoutMs = 30000) {
  // open recorder page which calls getUserMedia when user clicks its button
  const url = chrome.runtime.getURL('recorder.html');
  const win = window.open(url, 'whisper-permission', 'width=480,height=260');

  // Poll permission state / devices
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    // try Permissions API (best-effort)
    try {
      if (navigator.permissions && navigator.permissions.query) {
        try {
          const p = await navigator.permissions.query({ name: 'microphone' });
          if (p.state === 'granted') {
            if (win && !win.closed) win.close();
            return true;
          }
          // if denied, stop waiting
          if (p.state === 'denied') {
            if (win && !win.closed) win.close();
            return false;
          }
        } catch (e) {
          // ignore permissions.query errors in some browsers
        }
      }
    } catch (e) { /* ignore */ }

    // fallback: enumerate devices and check if labels are present
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs = devices.filter(d => d.kind === 'audioinput');
      if (inputs.length && inputs.some(d => d.label && d.label.length > 0)) {
        if (win && !win.closed) win.close();
        return true;
      }
    } catch (e) {
      // enumerateDevices might throw if no permission — ignore and continue polling
    }

    // wait a bit before next check
    await new Promise(res => setTimeout(res, 1000));
  }

  // timeout
  try { if (win && !win.closed) win.close(); } catch(e){}
  return false;
}

/**
 * getMicrophoneStream from earlier, with addition:
 * if permission's 'prompt' and we get NotAllowedError, open permission window and retry once.
 */
async function getMicrophoneStream(opts = {}) {
  const {
    preferredDeviceLabelRegex = null,
    preferredDeviceId = null,
    sampleRate = undefined,
    channelCount = undefined,
    echoCancellation = true,
    noiseSuppression = true,
    promptIfNeeded = true,
  } = opts;

  const diagnostics = [];
  let permissionState = 'unknown';

  // Permission query
  try {
    if (navigator.permissions && navigator.permissions.query) {
      try {
        const p = await navigator.permissions.query({ name: 'microphone' });
        permissionState = p.state; // "granted", "denied", "prompt"
        diagnostics.push(`Permissions API: microphone state=${permissionState}`);
      } catch (e) {
        diagnostics.push(`Permissions API: query unsupported or failed: ${e.name || e}`);
      }
    } else {
      diagnostics.push('Permissions API not available');
    }
  } catch (e) {
    diagnostics.push('Permissions query failed: ' + (e.message || e.name || e));
  }

  async function listAudioInputs() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs = devices.filter(d => d.kind === 'audioinput');
      diagnostics.push(`enumerateDevices: found ${inputs.length} audioinput(s)`);
      return inputs;
    } catch (e) {
      diagnostics.push(`enumerateDevices failed: ${e.name || e}`);
      return [];
    }
  }

  let devices = await listAudioInputs();

  // If no devices discovered yet, do a minimal prompt to reveal devices (best-effort)
  if (devices.length === 0 && promptIfNeeded) {
    diagnostics.push('No audioinput found initially — attempting a minimal getUserMedia prompt to discover devices.');
    try {
      const tmpStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      tmpStream.getTracks().forEach(t => t.stop());
      devices = await listAudioInputs();
      diagnostics.push(`Re-enumeration after prompt: found ${devices.length} audioinput(s)`);
    } catch (e) {
      diagnostics.push(`Minimal prompt failed: ${e.name || e}.`);
    }
  }

  // If permission state is prompt, proactively open permission window to let user accept
  if (permissionState === 'prompt') {
    diagnostics.push('Permissions state is prompt; opening permission window to ask user to accept.');
    const granted = await openPermissionWindowAndWait(30000);
    if (!granted) {
      diagnostics.push('Permission window timed out or was denied.');
      const err = new Error('Permission not granted by user.');
      err.diagnostics = diagnostics;
      throw err;
    }
    // re-evaluate permission and devices after user acceptance
    diagnostics.push('Permission window indicated granted; re-enumerating devices.');
    try {
      const p2 = await navigator.permissions.query({ name: 'microphone' });
      diagnostics.push(`Permissions API after grant: ${p2.state}`);
      permissionState = p2.state;
    } catch(e){}
    devices = await listAudioInputs();
  }

  // choose device
  let chosenDevice = null;
  if (preferredDeviceId && devices.length) {
    chosenDevice = devices.find(d => d.deviceId === preferredDeviceId) || null;
    if (chosenDevice) diagnostics.push(`Preferred deviceId matched: ${chosenDevice.label || chosenDevice.deviceId}`);
    else diagnostics.push(`Preferred deviceId not found: ${preferredDeviceId}`);
  }

  if (!chosenDevice && preferredDeviceLabelRegex && devices.length) {
    chosenDevice = devices.find(d => (d.label || '').match(preferredDeviceLabelRegex)) || null;
    if (chosenDevice) diagnostics.push(`Chosen by label regex: ${chosenDevice.label || chosenDevice.deviceId}`);
    else diagnostics.push(`No device label matched regex ${preferredDeviceLabelRegex}`);
  }

  if (!chosenDevice && devices.length) {
    chosenDevice = devices[0];
    diagnostics.push(`Falling back to first device: ${chosenDevice.label || chosenDevice.deviceId}`);
  }

  // constraint attempts
  const attempts = [];
  if (chosenDevice && chosenDevice.deviceId) {
    const c = {
      sampleRate,
      channelCount,
      echoCancellation,
      noiseSuppression,
      deviceId: { exact: chosenDevice.deviceId }
    };
    attempts.push({ name: 'device+advanced-constraints', constraints: { audio: c } });
  }
  attempts.push({ name: 'advanced-constraints', constraints: { audio: { sampleRate, channelCount, echoCancellation, noiseSuppression } } });
  attempts.push({ name: 'simple-audio-true', constraints: { audio: true } });
  attempts.push({ name: 'no-aec-noise', constraints: { audio: { echoCancellation: false, noiseSuppression: false } } });

  let lastError = null;
  for (const a of attempts) {
    const c = a.constraints;
    diagnostics.push(`Attempting constraints: ${a.name} => ${JSON.stringify(c)}`);
    try {
      const stream = await navigator.mediaDevices.getUserMedia(c);
      const track = stream.getAudioTracks()[0];
      const settings = track && track.getSettings ? track.getSettings() : null;
      const deviceInfo = {
        label: chosenDevice ? chosenDevice.label : (devices[0] && devices[0].label) || '',
        deviceId: settings && settings.deviceId ? settings.deviceId : (chosenDevice && chosenDevice.deviceId) || null,
        settings
      };
      diagnostics.push(`getUserMedia success with ${a.name}`);
      return { stream, deviceInfo, constraintsUsed: c, diagnostics };
    } catch (e) {
      lastError = e;
      diagnostics.push(`getUserMedia failed for ${a.name}: ${e.name || e}`);
      // If permission denied, we should not continue attempts; but if permission was prompt-> user denied we might have just caught that.
      if (e && (e.name === 'NotAllowedError' || e.name === 'SecurityError' || e.name === 'PermissionDeniedError')) {
        diagnostics.push('Permission denied — stopping further attempts.');
        break;
      }
      if (e && (e.name === 'NotFoundError' || e.name === 'DevicesNotFoundError')) {
        diagnostics.push('Device not found. Will continue with simpler constraints if available.');
      }
    }
  }

  const errorSummary = lastError ? `${lastError.name || lastError}` : 'UnknownError';
  const err = new Error(`Unable to obtain microphone stream: ${errorSummary}`);
  err.diagnostics = diagnostics;
  err.lastError = lastError;
  throw err;
}

// --- VAD + MediaRecorder logic remains same as before, using getMicrophoneStream() ---
async function startListening() {
  setStatus('requesting microphone...');
  try {
    const result = await getMicrophoneStream({
      preferredDeviceLabelRegex: /Microphone|USB|Internal/i,
      sampleRate: 48000,
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true
    });

    audioStream = result.stream;
    console.log('Microphone acquisition diagnostics:\n' + result.diagnostics.join('\n'));
    transcriptEl.textContent = ''; // clear any previous diagnostics shown to user

  } catch (e) {
    setStatus('microphone access denied');
    const diag = e && e.diagnostics ? e.diagnostics.join('\n') : (e.message || e.toString());
    transcriptEl.textContent = 'Error acquiring microphone.\n\nDiagnostics:\n' + diag;
    console.error('Microphone acquisition failed:', e);
    return;
  }

  // Setup analyser for VAD
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(audioStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  src.connect(analyser);
  const bufferLength = analyser.fftSize;
  dataArray = new Float32Array(bufferLength);

  // Setup MediaRecorder but only start recording when voice detected
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ?
               'audio/webm;codecs=opus' : 'audio/webm';
  try {
    mediaRecorder = new MediaRecorder(audioStream, { mimeType: mime });
  } catch (err) {
    mediaRecorder = new MediaRecorder(audioStream);
  }

  mediaRecorder.ondataavailable = e => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };
  mediaRecorder.onstop = onRecorderStop;

  startDetectedSince = null;
  silenceStartedAt = null;
  isRecording = false;

  // Start monitoring loop
  monitoringTimer = setInterval(monitorAudio, 100);
  setStatus('listening for voice (will auto-record)');
  listenBtn.disabled = true;
  stopBtn.disabled = false;
}

function stopListening() {
  if (monitoringTimer) {
    clearInterval(monitoringTimer);
    monitoringTimer = null;
  }
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
  }
  if (audioStream) {
    audioStream.getTracks().forEach(t => t.stop());
    audioStream = null;
  }
  if (audioCtx) {
    try { audioCtx.close(); } catch(e) {}
    audioCtx = null;
  }
  listenBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus('idle');
}

function rmsFromFloat32(arr) {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    sum += v * v;
  }
  return Math.sqrt(sum / arr.length);
}

function monitorAudio() {
  if (!analyser) return;
  analyser.getFloatTimeDomainData(dataArray);
  const rms = rmsFromFloat32(dataArray);

  if (!isRecording) {
    if (rms >= START_RMS) {
      if (!startDetectedSince) startDetectedSince = Date.now();
      if (Date.now() - startDetectedSince >= START_PERSIST_MS) {
        startRecording();
      }
    } else {
      startDetectedSince = null;
    }
  } else {
    if (rms < SILENCE_RMS) {
      if (!silenceStartedAt) silenceStartedAt = Date.now();
      else if (Date.now() - silenceStartedAt >= SILENCE_TIMEOUT_MS) {
        try { mediaRecorder.stop(); } catch(e){ console.warn('mediaRecorder.stop error', e); }
        silenceStartedAt = null;
      }
    } else {
      silenceStartedAt = null;
    }
  }
}

function startRecording() {
  chunks = [];
  try {
    mediaRecorder.start();
  } catch (e) {
    console.warn('mediaRecorder.start failed, trying again', e);
    mediaRecorder.start();
  }
  isRecording = true;
  setStatus('recording...');
}


async function onRecorderStop() {
  isRecording = false;
  setStatus('uploading...');
  const blob = new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' });
  const url = URL.createObjectURL(blob);
  transcriptEl.textContent = 'Processing audio...';

  // Helper: render filename pattern
  function renderFilename(pattern) {
    const now = new Date();
    const pad = n => String(n).padStart(2,'0');
    const date = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}`;
    const time = `${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
    const timestamp = String(now.getTime());
    const seq = Math.floor(Math.random()*1000000);
    // sanitize pattern a bit
    let out = (pattern || 'whisper/recording-{date}-{time}.webm');
    out = out.replaceAll('{date}', date)
             .replaceAll('{time}', time)
             .replaceAll('{timestamp}', timestamp)
             .replaceAll('{seq}', seq);
    // Replace windows backslashes with forward slashes and avoid leading /
    out = out.replace(/\\/g, '/').replace(/^\/+/, '');
    return out;
  }

  // 1) Save locally to Downloads if enabled
  if (extOptions.saveLocal) {
    try {
      const filename = renderFilename(extOptions.filenamePattern);
      // chrome.downloads.download expects a URL (object URL) and filename relative to Downloads
      chrome.downloads.download({
        url,
        filename,
        conflictAction: 'uniquify', // avoid overwriting
        saveAs: false
      }, downloadId => {
        if (chrome.runtime.lastError) {
          console.warn('Download failed:', chrome.runtime.lastError);
          transcriptEl.textContent = 'Save failed: ' + chrome.runtime.lastError.message;
        } else {
          transcriptEl.textContent = `Saved to Downloads/${filename}`;
        }
        // revoke object URL after a small delay to ensure download initiated
        setTimeout(() => URL.revokeObjectURL(url), 3000);
      });
    } catch (e) {
      console.error('Saving locally failed', e);
      transcriptEl.textContent = 'Save failed: ' + (e.message || e.name);
      // ensure object URL revoked
      URL.revokeObjectURL(url);
    }
  } else {
    // If not saving locally, revoke right away (we may still upload)
    URL.revokeObjectURL(url);
  }

  // 2) Upload to server if configured
  if (extOptions.alsoUpload) {
    try {
      setStatus('uploading...');
      const fd = new FormData();
      fd.append('file', blob, 'recording.webm');
      const res = await fetch('http://localhost:5000/transcribe', {
        method: 'POST',
        body: fd
      });

      if (!res.ok) {
        const txt = await res.text();
        setStatus('server error');
        transcriptEl.textContent = 'Error: ' + txt;
        return;
      }

      const data = await res.json();
      setStatus('transcription received');
      // If also saved locally, append the transcription to the status area
      if (extOptions.saveLocal) {
        transcriptEl.textContent += '\nTranscription: ' + (data.text || '(no text)');
      } else {
        transcriptEl.textContent = data.text || '(no text)';
      }
    } catch (e) {
      console.error(e);
      setStatus('upload failed');
      transcriptEl.textContent = 'Upload error: ' + e.message;
    } finally {
      listenBtn.disabled = false;
      stopBtn.disabled = true;
    }
  } else {
    // If not uploading, finalize UI
    setStatus('idle');
    listenBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

/*
async function onRecorderStop() {
  isRecording = false;
  setStatus('uploading...');
  const blob = new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' });
  transcriptEl.textContent = 'Uploading audio...';

  try {
    const fd = new FormData();
    fd.append('file', blob, 'recording.webm');
    const res = await fetch('http://localhost:5000/transcribe', {
      method: 'POST',
      body: fd
    });

    if (!res.ok) {
      const txt = await res.text();
      setStatus('server error');
      transcriptEl.textContent = 'Error: ' + txt;
      return;
    }

    const data = await res.json();
    setStatus('transcription received');
    transcriptEl.textContent = data.text || '(no text)';
  } catch (e) {
    console.error(e);
    setStatus('upload failed');
    transcriptEl.textContent = 'Upload error: ' + e.message;
  } finally {
    listenBtn.disabled = false;
    stopBtn.disabled = true;
  }
}
*/

listenBtn.addEventListener('click', startListening);
stopBtn.addEventListener('click', stopListening);
