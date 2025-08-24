// recorder.js
(function() {
  const requestBtn = document.getElementById('requestBtn');
  const closeBtn = document.getElementById('closeBtn');
  const status = document.getElementById('status');

  requestBtn.addEventListener('click', async () => {
    status.textContent = 'Requesting microphone... (accept the browser prompt)';
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      // stop tracks immediately - we only asked to trigger/receive permission
      s.getTracks().forEach(t => t.stop());
      status.textContent = 'Permission granted. You can close this window.';
    } catch (e) {
      status.textContent = 'Request failed: ' + (e.name || e.message);
      console.error('recorder request error', e);
    }
  });

  closeBtn.addEventListener('click', () => window.close());
})();
