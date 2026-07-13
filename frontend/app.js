const API = "/api";

let voices = [];
let selectedVoiceId = null;
let recordedBlob = null;
let audioCtx = null;
let mediaStream = null;
let recordedChunks = [];
let isRecording = false;

const voiceListEl = document.getElementById("voice-list");
const voiceSelectEl = document.getElementById("voice-select");
const generateBtn = document.getElementById("generate-btn");
const generateStatusEl = document.getElementById("generate-status");
const resultEl = document.getElementById("result");
const audioPlayerEl = document.getElementById("audio-player");
const downloadLinkEl = document.getElementById("download-link");
const textInputEl = document.getElementById("text-input");

const modalEl = document.getElementById("clone-modal");
const newVoiceBtn = document.getElementById("new-voice-btn");
const closeModalBtn = document.getElementById("close-modal-btn");
const voiceNameInput = document.getElementById("voice-name-input");
const voiceFileInput = document.getElementById("voice-file-input");
const cloneStatusEl = document.getElementById("clone-status");
const submitCloneBtn = document.getElementById("submit-clone-btn");
const recordBtn = document.getElementById("record-btn");
const recordTimerEl = document.getElementById("record-timer");
const recordPreviewEl = document.getElementById("record-preview");

const settingsModalEl = document.getElementById("settings-modal");
const settingsBtn = document.getElementById("settings-btn");
const closeSettingsBtn = document.getElementById("close-settings-btn");
const settingsStatusBadgeEl = document.getElementById("settings-status-badge");
const settingsStatusEl = document.getElementById("settings-status");
const hfTokenInput = document.getElementById("hf-token-input");
const saveSettingsBtn = document.getElementById("save-settings-btn");
const geminiStatusBadgeEl = document.getElementById("gemini-status-badge");
const geminiKeyInput = document.getElementById("gemini-key-input");

const storyboardBtn = document.getElementById("storyboard-btn");
const storyboardStatusEl = document.getElementById("storyboard-status");

async function loadVoices() {
  const res = await fetch(`${API}/voices`);
  const data = await res.json();
  voices = data.voices;
  renderVoiceList();
  renderVoiceSelect();
}

function renderVoiceList() {
  voiceListEl.innerHTML = "";
  for (const voice of voices) {
    const item = document.createElement("div");
    item.className = "voice-item";
    item.innerHTML = `
      <span>${voice.name}</span>
      <span style="display:flex;align-items:center;gap:6px;">
        <span class="tag">${voice.type}</span>
        ${voice.type === "cloned" ? '<button class="delete-btn" title="Delete">🗑</button>' : ""}
      </span>
    `;
    if (voice.type === "cloned") {
      item.querySelector(".delete-btn").addEventListener("click", () => deleteVoice(voice.id));
    }
    voiceListEl.appendChild(item);
  }
}

function renderVoiceSelect() {
  const previous = voiceSelectEl.value;
  voiceSelectEl.innerHTML = "";
  for (const voice of voices) {
    const opt = document.createElement("option");
    opt.value = voice.id;
    opt.textContent = `${voice.name} (${voice.type})`;
    voiceSelectEl.appendChild(opt);
  }
  if (previous && voices.some((v) => v.id === previous)) {
    voiceSelectEl.value = previous;
  }
}

async function deleteVoice(voiceId) {
  if (!confirm("Delete this cloned voice?")) return;
  await fetch(`${API}/voices/${voiceId}`, { method: "DELETE" });
  await loadVoices();
}

generateBtn.addEventListener("click", async () => {
  const text = textInputEl.value.trim();
  const voiceId = voiceSelectEl.value;
  if (!text) {
    setStatus(generateStatusEl, "Please enter some text.", true);
    return;
  }
  if (!voiceId) {
    setStatus(generateStatusEl, "Please select a voice.", true);
    return;
  }

  generateBtn.disabled = true;
  setStatus(generateStatusEl, "Generating audio... this can take a few seconds.");
  resultEl.classList.add("hidden");

  try {
    const form = new FormData();
    form.append("text", text);
    form.append("voice_id", voiceId);
    const res = await fetch(`${API}/generate`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Generation failed.");
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    audioPlayerEl.src = url;
    downloadLinkEl.href = url;
    resultEl.classList.remove("hidden");
    setStatus(generateStatusEl, "Done.");
    audioPlayerEl.play();
  } catch (err) {
    setStatus(generateStatusEl, err.message, true);
  } finally {
    generateBtn.disabled = false;
  }
});

function setStatus(el, message, isError = false) {
  el.textContent = message;
  el.classList.toggle("error", isError);
}

// ---- Clone modal ----

newVoiceBtn.addEventListener("click", () => {
  modalEl.classList.remove("hidden");
});

closeModalBtn.addEventListener("click", closeModal);
modalEl.addEventListener("click", (e) => {
  if (e.target === modalEl) closeModal();
});

function closeModal() {
  modalEl.classList.add("hidden");
  stopRecording();
  recordedBlob = null;
  recordPreviewEl.classList.add("hidden");
  voiceNameInput.value = "";
  voiceFileInput.value = "";
  setStatus(cloneStatusEl, "");
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
  });
});

recordBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

async function startRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus(cloneStatusEl, "Microphone access denied.", true);
    return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(mediaStream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);
  recordedChunks = [];

  source.connect(processor);
  processor.connect(audioCtx.destination);
  processor.onaudioprocess = (e) => {
    recordedChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  audioCtx._processor = processor;
  audioCtx._source = source;

  isRecording = true;
  recordBtn.textContent = "■ Stop recording";
  let seconds = 0;
  recordTimerEl.textContent = "0:00";
  audioCtx._timerInterval = setInterval(() => {
    seconds += 1;
    const m = Math.floor(seconds / 60);
    const s = String(seconds % 60).padStart(2, "0");
    recordTimerEl.textContent = `${m}:${s}`;
  }, 1000);
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  recordBtn.textContent = "● Start recording";
  clearInterval(audioCtx._timerInterval);

  audioCtx._processor.disconnect();
  audioCtx._source.disconnect();
  mediaStream.getTracks().forEach((t) => t.stop());

  const sampleRate = audioCtx.sampleRate;
  recordedBlob = encodeWav(recordedChunks, sampleRate);
  recordPreviewEl.src = URL.createObjectURL(recordedBlob);
  recordPreviewEl.classList.remove("hidden");

  audioCtx.close();
  audioCtx = null;
}

function encodeWav(chunks, sampleRate) {
  const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
  const samples = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }

  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let pos = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(pos, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    pos += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

submitCloneBtn.addEventListener("click", async () => {
  const name = voiceNameInput.value.trim();
  if (!name) {
    setStatus(cloneStatusEl, "Please enter a voice name.", true);
    return;
  }

  const activeTab = document.querySelector(".tab-btn.active").dataset.tab;
  let file = null;
  if (activeTab === "upload") {
    file = voiceFileInput.files[0];
    if (!file) {
      setStatus(cloneStatusEl, "Please choose an audio file.", true);
      return;
    }
  } else {
    if (!recordedBlob) {
      setStatus(cloneStatusEl, "Please record a sample first.", true);
      return;
    }
    file = new File([recordedBlob], "recording.wav", { type: "audio/wav" });
  }

  submitCloneBtn.disabled = true;
  setStatus(cloneStatusEl, "Cloning voice... this can take a moment.");

  try {
    const form = new FormData();
    form.append("name", name);
    form.append("file", file);
    const res = await fetch(`${API}/voices`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Cloning failed.");
    }
    const voice = await res.json();
    await loadVoices();
    voiceSelectEl.value = voice.id;
    closeModal();
  } catch (err) {
    setStatus(cloneStatusEl, err.message, true);
  } finally {
    submitCloneBtn.disabled = false;
  }
});

// ---- Settings modal ----

settingsBtn.addEventListener("click", () => {
  settingsModalEl.classList.remove("hidden");
  loadSettingsStatus();
});

closeSettingsBtn.addEventListener("click", () => {
  settingsModalEl.classList.add("hidden");
  hfTokenInput.value = "";
  setStatus(settingsStatusEl, "");
});

settingsModalEl.addEventListener("click", (e) => {
  if (e.target === settingsModalEl) closeSettingsBtn.click();
});

async function loadSettingsStatus() {
  const res = await fetch(`${API}/settings`);
  const data = await res.json();
  renderSettingsBadge(data);
}

function renderSettingsBadge(data) {
  if (data.hf_token_configured) {
    settingsStatusBadgeEl.textContent = `✓ Connected as ${data.hf_username || "unknown user"}`;
    settingsStatusBadgeEl.className = "badge ok";
  } else {
    settingsStatusBadgeEl.textContent = "Not configured — using built-in voices only";
    settingsStatusBadgeEl.className = "badge missing";
  }

  if (data.gemini_api_key_configured) {
    geminiStatusBadgeEl.textContent = "✓ Gemini API key configured";
    geminiStatusBadgeEl.className = "badge ok";
  } else {
    geminiStatusBadgeEl.textContent = "Not configured — photo sequence generation disabled";
    geminiStatusBadgeEl.className = "badge missing";
  }
}

saveSettingsBtn.addEventListener("click", async () => {
  const hfToken = hfTokenInput.value.trim();
  const geminiKey = geminiKeyInput.value.trim();
  if (!hfToken && !geminiKey) {
    setStatus(settingsStatusEl, "Please paste at least one token/key.", true);
    return;
  }
  saveSettingsBtn.disabled = true;
  setStatus(settingsStatusEl, "Saving...");
  try {
    const payload = {};
    if (hfToken) payload.hf_token = hfToken;
    if (geminiKey) payload.gemini_api_key = geminiKey;
    const res = await fetch(`${API}/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Could not save settings.");
    renderSettingsBadge(data);
    setStatus(settingsStatusEl, "Saved.");
    hfTokenInput.value = "";
    geminiKeyInput.value = "";
  } catch (err) {
    setStatus(settingsStatusEl, err.message, true);
  } finally {
    saveSettingsBtn.disabled = false;
  }
});

// ---- Storyboard (Gemini photo sequence) ----

storyboardBtn.addEventListener("click", async () => {
  const text = textInputEl.value.trim();
  if (!text) {
    setStatus(storyboardStatusEl, "Please enter some text first.", true);
    return;
  }

  storyboardBtn.disabled = true;
  setStatus(storyboardStatusEl, "Generating matching photos with Gemini... this can take a while.");

  try {
    const form = new FormData();
    form.append("text", text);
    const res = await fetch(`${API}/storyboard`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Photo generation failed.");
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "storyboard.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setStatus(storyboardStatusEl, "Done — storyboard.zip downloaded.");
  } catch (err) {
    setStatus(storyboardStatusEl, err.message, true);
  } finally {
    storyboardBtn.disabled = false;
  }
});

loadVoices();
