const recordButton = document.getElementById("recordButton");
const recordStatus = document.getElementById("recordStatus");
const logOutput = document.getElementById("logOutput");
const resultsPanel = document.getElementById("resultsPanel");
const transcriptText = document.getElementById("transcriptText");
const languageInfo = document.getElementById("languageInfo");
const tokenCount = document.getElementById("tokenCount");
const tokenList = document.getElementById("tokenList");
const summaryBlock = document.getElementById("summaryBlock");
const summaryText = document.getElementById("summaryText");
const answerText = document.getElementById("answerText");
const downloadBtn = document.getElementById("downloadBtn");

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let lastResultText = "";

function appendLog(message) {
  const timestamp = new Date().toLocaleTimeString();
  logOutput.textContent += `[${timestamp}] ${message}\n`;
  logOutput.scrollTop = logOutput.scrollHeight;
}

async function ensureRecorder() {
  if (mediaRecorder) return;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.addEventListener("dataavailable", (event) => {
    audioChunks.push(event.data);
  });
  mediaRecorder.addEventListener("stop", async () => {
    recordButton.classList.remove("active");
    recordButton.classList.add("idle");
    recordStatus.textContent = "Processing...";
    appendLog("Recording finished. Preparing upload...");
    const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
    audioChunks = [];
    await uploadRecording(blob);
  });
}

async function uploadRecording(blob) {
  try {
    const formData = new FormData();
    formData.append("file", blob, "recording.webm");

    const provider = document.querySelector("input[name='provider']:checked").value;
    formData.append("provider", provider);
    formData.append("openai_model", document.getElementById("openaiModel").value.trim());
    formData.append("whisper_model", document.getElementById("whisperModel").value.trim());
    formData.append("tokenizer_model", document.getElementById("tokenizerModel").value.trim());
    formData.append("encoding_name", document.getElementById("encodingName").value.trim());

    const shouldSummarize = document.getElementById("shouldSummarize").checked;
    formData.append("summarize", shouldSummarize);
    formData.append("ollama_model", document.getElementById("ollamaModel").value.trim());
    formData.append("ollama_url", document.getElementById("ollamaUrl").value.trim());

    appendLog("Uploading audio...");
    const response = await fetch("/transcribe", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Server error: ${response.status} ${text}`);
    }

    const payload = await response.json();
    appendLog("Transcription complete.");
    displayResults(payload);
  } catch (error) {
    appendLog(`Error: ${error.message}`);
    recordStatus.textContent = "Tap to start recording";
  }
}

function displayResults(payload) {
  lastResultText = buildReport(payload);
  resultsPanel.classList.remove("hide");

  transcriptText.textContent = payload.transcript ?? "";
  languageInfo.textContent = payload.language ? `Language: ${payload.language}` : "";
  tokenCount.textContent = `Tokens (${payload.encoding_name}): ${payload.token_count}`;
  tokenList.textContent = (payload.tokens || []).join(", ");

  if (payload.summary) {
    summaryBlock.classList.remove("hide");
    summaryText.textContent = payload.summary;
    answerText.textContent = payload.answer ? `Answer: ${payload.answer}` : "";
  } else {
    summaryBlock.classList.add("hide");
  }

  recordStatus.textContent = "Tap to start recording";
}

function buildReport(payload) {
  const chunks = [
    "=== Transcription Result ===",
    `Text: ${payload.transcript ?? ""}`,
    payload.language ? `Language: ${payload.language}` : null,
    `Token count (${payload.encoding_name}): ${payload.token_count}`,
  ];

  if (payload.summary) {
    chunks.push(`Summary:\n${payload.summary}`);
  }
  if (payload.answer) {
    chunks.push(`Answer: ${payload.answer}`);
  }

  return chunks.filter(Boolean).join("\n");
}

recordButton.addEventListener("click", async () => {
  try {
    await ensureRecorder();
    if (!isRecording) {
      audioChunks = [];
      mediaRecorder.start();
      isRecording = true;
      recordButton.classList.remove("idle");
      recordButton.classList.add("active");
      recordStatus.textContent = "Recording... tap to stop";
      appendLog("Recording started.");
    } else {
      mediaRecorder.stop();
      isRecording = false;
      appendLog("Stopping recorder...");
    }
  } catch (error) {
    appendLog(`Recording error: ${error.message}`);
  }
});

downloadBtn.addEventListener("click", () => {
  if (!lastResultText) {
    appendLog("Nothing to download yet.");
    return;
  }
  const blob = new Blob([lastResultText], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `transcript_${Date.now()}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
});

appendLog("Ready. Tap the red button to record.");

