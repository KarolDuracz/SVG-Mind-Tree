// options.js
const saveLocalEl = document.getElementById('saveLocal');
const filenamePatternEl = document.getElementById('filenamePattern');
const alsoUploadEl = document.getElementById('alsoUpload');
const saveBtn = document.getElementById('saveBtn');
const defaultBtn = document.getElementById('defaultBtn');
const previewEl = document.getElementById('preview');

const DEFAULTS = {
  saveLocal: true,
  filenamePattern: 'whisper/recording-{date}-{time}.webm',
  alsoUpload: true
};

function fillPreview() {
  const pattern = filenamePatternEl.value || DEFAULTS.filenamePattern;
  previewEl.textContent = formatPattern(pattern);
}

function formatPattern(pattern) {
  const now = new Date();
  const pad = n => String(n).padStart(2,'0');
  const date = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}`;
  const time = `${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
  const timestamp = String(now.getTime());
  const seq = Math.floor(Math.random()*10000);
  return pattern
    .replaceAll('{date}', date)
    .replaceAll('{time}', time)
    .replaceAll('{timestamp}', timestamp)
    .replaceAll('{seq}', seq);
}

function saveOptions() {
  const opts = {
    saveLocal: saveLocalEl.checked,
    filenamePattern: filenamePatternEl.value || DEFAULTS.filenamePattern,
    alsoUpload: alsoUploadEl.checked
  };
  chrome.storage.sync.set(opts, () => {
    saveBtn.textContent = 'Saved';
    setTimeout(()=> saveBtn.textContent = 'Save', 1000);
  });
  fillPreview();
}

function restoreDefaults() {
  saveLocalEl.checked = DEFAULTS.saveLocal;
  filenamePatternEl.value = DEFAULTS.filenamePattern;
  alsoUploadEl.checked = DEFAULTS.alsoUpload;
  fillPreview();
  chrome.storage.sync.set(DEFAULTS);
}

function loadOptions() {
  chrome.storage.sync.get(DEFAULTS, (items) => {
    saveLocalEl.checked = items.saveLocal;
    filenamePatternEl.value = items.filenamePattern;
    alsoUploadEl.checked = items.alsoUpload;
    fillPreview();
  });
}

saveBtn.addEventListener('click', saveOptions);
defaultBtn.addEventListener('click', restoreDefaults);
filenamePatternEl.addEventListener('input', fillPreview);

loadOptions();
