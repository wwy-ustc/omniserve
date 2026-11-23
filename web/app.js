"use strict";

const find = (id) => document.getElementById(id);
const statusLabels = {
  queued: "等待转写",
  transcribing: "转写中",
  waiting: "等待生成",
  generating: "生成中",
  done: "已完成",
  error: "失败",
};
const terminal = new Set(["done", "error"]);
const state = { jobs: [], now: 0, selected: null, historyKey: "", timelineKey: "" };
let recording = null;
// Leave room for encoder padding below the server's 30-second audio limit.
const MAX_RECORDING_MS = 29000;

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}

function updateText(node, value) {
  if (node.textContent !== value) node.textContent = value;
}

function notice(message = "", kind = "info") {
  const node = find("notice");
  node.textContent = message;
  node.hidden = !message;
  node.dataset.kind = kind;
}

function statusBadge(node, status) {
  node.dataset.status = status;
  updateText(node, statusLabels[status] || status);
}

function jobName(job) {
  return job.name || job.id;
}

function makeSelect(job, className) {
  const button = element("button", className);
  button.type = "button";
  button.dataset.job = job.id;
  button.title = jobName(job);
  button.setAttribute("aria-pressed", String(job.id === state.selected));
  button.addEventListener("click", () => selectJob(job.id));
  return button;
}

function selectJob(id) {
  state.selected = id;
  for (const button of document.querySelectorAll("button[data-job]")) {
    button.setAttribute("aria-pressed", String(button.dataset.job === id));
  }
  renderDetail();
}

function replaceWithFocus(container, children) {
  const focused = container.contains(document.activeElement)
    ? document.activeElement.dataset.job : null;
  const scrollTop = container.scrollTop;
  container.replaceChildren(...children);
  container.scrollTop = scrollTop;
  if (focused) {
    const button = Array.from(container.querySelectorAll("button[data-job]"))
      .find((node) => node.dataset.job === focused);
    button?.focus({ preventScroll: true });
  }
}

function renderHistory() {
  const jobs = state.jobs;
  const historyKey = JSON.stringify(jobs.map(({ id, name, status }) => [id, name, status]));
  if (state.historyKey === historyKey) return;
  state.historyKey = historyKey;
  find("history-empty").hidden = jobs.length > 0;
  updateText(find("request-count"), `${jobs.length} 条`);
  const rows = jobs.map((job) => {
    const row = element("li");
    const button = makeSelect(job, "job-button");
    const badge = element("span", "badge");
    statusBadge(badge, job.status);
    button.append(element("span", "job-name", jobName(job)), badge);
    row.append(button);
    return row;
  });
  replaceWithFocus(find("history"), rows);
}

function formatDuration(ms) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

function renderDetail() {
  const job = state.jobs.find(({ id }) => id === state.selected);
  for (const id of ["detail-name", "detail-status", "detail-content"]) find(id).hidden = !job;
  find("detail-empty").hidden = Boolean(job);
  if (!job) return;

  updateText(find("detail-name"), jobName(job));
  statusBadge(find("detail-status"), job.status);
  const out = terminal.has(job.status) ? job.finished_ms : state.now;
  const timings = [
    ["转写", job.asr_started_ms, job.asr_finished_ms || out],
    ["生成", job.llm_started_ms, out],
    ["总耗时", job.created_ms, out],
  ].filter(([, start, end]) => Number.isFinite(start) && Number.isFinite(end));
  find("detail-times").replaceChildren(...timings.map(([label, start, end]) => {
    const item = element("div");
    item.append(element("dt", null, label), element("dd", null, formatDuration(Math.max(0, end - start))));
    return item;
  }));

  const transcript = find("transcript");
  const answer = find("answer");
  const failed = job.status === "error";
  updateText(transcript, job.transcript || (failed ? "无转写结果" : "等待语音转写…"));
  updateText(answer, job.answer || (terminal.has(job.status) ? "无文本回答" : "等待模型生成…"));
  transcript.dataset.placeholder = String(!job.transcript);
  answer.dataset.placeholder = String(!job.answer);
  find("error").hidden = !job.error;
  updateText(find("error"), job.error || "");
}

function stageSpans(job) {
  const { created_ms, asr_started_ms, asr_finished_ms, llm_started_ms } = job;
  const end = job.finished_ms ?? state.now;
  return [
    ["queue", "等待转写", created_ms, asr_started_ms ?? end],
    ["asr", "语音转写", asr_started_ms, asr_finished_ms ?? end],
    ["queue", "等待生成", asr_finished_ms, llm_started_ms ?? end],
    ["llm", "模型生成", llm_started_ms, end],
  ].filter(([, , start, stop]) => Number.isFinite(start) && Number.isFinite(stop) && stop > start);
}

function longestOverlap(timelines) {
  let longest = 0;
  for (const [transcribingJob, spans] of timelines) {
    const transcribing = spans.find(([stage]) => stage === "asr");
    if (!transcribing) continue;
    for (const [generatingJob, otherSpans] of timelines) {
      if (generatingJob.id === transcribingJob.id) continue;
      const generating = otherSpans.find(([stage]) => stage === "llm");
      if (!generating) continue;
      const start = Math.max(transcribing[2], generating[2]);
      const end = Math.min(transcribing[3], generating[3]);
      longest = Math.max(longest, end - start);
    }
  }
  return longest;
}

function renderTimeline() {
  const jobs = state.jobs.slice(0, 12).reverse();
  find("timeline-empty").hidden = jobs.length > 0;
  find("timeline-scroll").hidden = jobs.length === 0;
  if (!jobs.length) {
    find("overlap").hidden = true;
    find("timeline").replaceChildren();
    state.timelineKey = "";
    return;
  }

  // Epoch times are aligned to the first submitted request on this shared axis.
  const timelines = jobs.map((job) => [job, stageSpans(job)]);
  const timelineKey = JSON.stringify(timelines.map(([job, spans]) => [job.id, job.name, spans]));
  if (state.timelineKey === timelineKey) return;
  state.timelineKey = timelineKey;
  const overlap = longestOverlap(timelines);
  const overlapLabel = find("overlap");
  overlapLabel.hidden = overlap <= 0;
  updateText(overlapLabel, overlap > 0 ? `最长一次转写与生成重叠 ${formatDuration(overlap)}` : "");
  const zero = Math.min(...jobs.map(({ created_ms }) => created_ms));
  const end = Math.max(...jobs.map(({ finished_ms }) => finished_ms ?? state.now));
  const step = Math.max(250, Math.ceil((end - zero) / 4000) * 1000);
  const width = step * 4;
  const axis = element("div", "axis");
  axis.append(...Array.from({ length: 5 }, (_, index) =>
    element("span", null, index === 0 ? "0" : `${(index * step) / 1000}s`)));
  const rows = timelines.flatMap(([job, spans]) => {
    const label = makeSelect(job, "row-label");
    label.textContent = jobName(job);
    const track = element("div", "track");
    track.append(...spans.map(([category, title, start, stop]) => {
      const bar = element("span", `segment ${category}`);
      bar.style.left = `${((start - zero) / width) * 100}%`;
      bar.style.width = `${((stop - start) / width) * 100}%`;
      bar.title = `${title} · ${formatDuration(stop - start)}`;
      bar.setAttribute("aria-label", bar.title);
      return bar;
    }));
    return [label, track];
  });
  replaceWithFocus(find("timeline"), [element("div", "axis-label", "最近请求"), axis, ...rows]);
}

function setJobState(jobs, now) {
  state.now = now;
  state.jobs = jobs.sort((left, right) => right.created_ms - left.created_ms);
  if (!jobs.some(({ id }) => id === state.selected)) state.selected = jobs[0]?.id ?? null;
  renderHistory();
  renderTimeline();
  renderDetail();
}

function deviceLabel(device) {
  const name = String(device || "").toLowerCase();
  if (name.startsWith("cuda")) return "CUDA 生成";
  if (name.startsWith("mps")) return "Apple MPS 生成";
  if (name.startsWith("cpu")) return "CPU 生成";
  return device ? `${device} 生成` : "等待推理设备";
}

async function refresh() {
  const connection = find("connection");
  try {
    const response = await fetch("/api/jobs", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { jobs, now_ms: now, device } = await response.json();
    setJobState(jobs, now);
    updateText(connection, "后端已连接");
    connection.dataset.state = "online";
    updateText(find("device"), deviceLabel(device));
  } catch (_) {
    updateText(connection, "后端连接中断");
    connection.dataset.state = "offline";
  } finally {
    window.setTimeout(refresh, 400);
  }
}

async function uploadAudio(blob, name) {
  const params = new URLSearchParams({ name });
  const result = await fetch(`/api/jobs?${params}`, {
    method: "POST",
    headers: { "Content-Type": blob.type || "application/octet-stream" },
    body: blob,
  });
  if (!result.ok) {
    const details = await result.json().catch(() => ({}));
    throw new Error(details.error || `上传失败（HTTP ${result.status}）`);
  }
  return result.json();
}

async function submitFiles(files) {
  if (!files.length) return;
  notice(`正在提交 ${files.length} 条语音…`);
  const results = await Promise.allSettled(files.map(({ blob, name }) => uploadAudio(blob, name)));
  const failures = [];
  let uploaded = 0;
  results.forEach((result, index) => {
    if (result.status === "fulfilled") uploaded += 1;
    else failures.push(`${files[index].name}：${result.reason.message}`);
  });
  if (failures.length) {
    notice(`已提交 ${uploaded} 条。${failures.join("；")}`, "error");
  } else {
    notice(`已提交 ${uploaded} 条语音。`);
  }
}

function submitSelectedFiles(files) {
  return submitFiles(Array.from(files, (file) => ({ blob: file, name: file.name })));
}

const fileInput = find("files");
find("upload").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  submitSelectedFiles(fileInput.files);
  fileInput.value = "";
});

const dropTarget = fileInput.closest(".upload-panel");
dropTarget.addEventListener("dragover", (event) => {
  if (!event.dataTransfer.types.includes("Files")) return;
  event.preventDefault();
  dropTarget.classList.add("dragging");
});
dropTarget.addEventListener("dragleave", (event) => {
  if (!dropTarget.contains(event.relatedTarget)) dropTarget.classList.remove("dragging");
});
dropTarget.addEventListener("drop", (event) => {
  dropTarget.classList.remove("dragging");
  if (!event.dataTransfer.files.length) return;
  event.preventDefault();
  submitSelectedFiles(event.dataTransfer.files);
});

const recordButton = find("record");
const recordLabel = find("record-label");

function setRecordingState(active) {
  recordButton.setAttribute("aria-pressed", String(active));
  recordLabel.textContent = active ? "结束并提交" : "开始录音";
}

recordButton.addEventListener("click", async () => {
  if (recording && recording.state !== "inactive") {
    recordButton.disabled = true;
    recording.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    notice("当前浏览器无法访问麦克风。你可以上传音频文件。", "error");
    return;
  }
  let stream;
  recordButton.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks = [];
    let failed = false;
    let stopTimer;
    recorder.addEventListener("dataavailable", ({ data }) => {
      if (data.size > 0) chunks.push(data);
    });
    recorder.addEventListener("error", () => {
      failed = true;
      notice("录音中断，请重试或上传音频文件。", "error");
    });
    recorder.addEventListener("stop", () => {
      window.clearTimeout(stopTimer);
      stream.getTracks().forEach((track) => track.stop());
      recording = null;
      recordButton.disabled = false;
      setRecordingState(false);
      if (failed) return;
      const type = recorder.mimeType || chunks[0]?.type || "application/octet-stream";
      const blob = new Blob(chunks, { type });
      if (!blob.size) {
        notice("未录到音频，请重试。", "error");
        return;
      }
      const extension = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
      submitFiles([{ blob, name: `录音 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}.${extension}` }]);
    }, { once: true });
    recorder.start();
    stopTimer = window.setTimeout(() => {
      if (recorder.state !== "inactive") {
        recordButton.disabled = true;
        recorder.stop();
      }
    }, MAX_RECORDING_MS);
    recording = recorder;
    setRecordingState(true);
    notice("正在录音，点「结束并提交」发送；约 29 秒后自动提交。");
  } catch (_) {
    stream?.getTracks().forEach((track) => track.stop());
    notice("无法开始录音，请检查麦克风权限或上传音频。", "error");
  } finally {
    recordButton.disabled = false;
  }
});

refresh();
