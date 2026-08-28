// Minimal viewer logic: paste a detection JSON response, point at a video
// URL, and see ad segments drawn on a seek bar + listed below. Clicking a
// segment seeks the video to that timestamp. Intentionally simple per the
// assignment brief ("do not spend more than an hour here").

const player = document.getElementById("player");
const seekbar = document.getElementById("seekbar");
const segmentList = document.getElementById("segmentList");
const videoUrlInput = document.getElementById("videoUrl");
const jsonInput = document.getElementById("jsonInput");
const loadBtn = document.getElementById("loadBtn");

let currentData = null;

loadBtn.addEventListener("click", () => {
  const videoUrl = videoUrlInput.value.trim();
  if (videoUrl) {
    player.src = videoUrl;
  }

  let data;
  try {
    data = JSON.parse(jsonInput.value);
  } catch (e) {
    alert("Invalid JSON: " + e.message);
    return;
  }
  currentData = data;
  render(data);
});

function render(data) {
  seekbar.innerHTML = "";
  segmentList.innerHTML = "";

  const duration = (data.source && data.source.duration_s) || 1;
  const segments = data.segments || [];

  segments.forEach((seg) => {
    const left = (seg.start_s / duration) * 100;
    const width = Math.max(0.3, ((seg.end_s - seg.start_s) / duration) * 100);

    const marker = document.createElement("div");
    marker.className = "segment-marker";
    marker.style.left = left + "%";
    marker.style.width = width + "%";
    marker.title = `${seg.ad_type} (${seg.start_s.toFixed(1)}s - ${seg.end_s.toFixed(1)}s)`;
    marker.addEventListener("click", () => seekTo(seg.start_s));
    seekbar.appendChild(marker);

    const card = document.createElement("div");
    card.className = "segment-card";
    card.addEventListener("click", () => seekTo(seg.start_s));
    card.innerHTML = `
      <div class="meta">
        <span class="badge">${seg.ad_type}</span>
        <span class="badge">conf ${seg.confidence.toFixed(2)}</span>
        ${seg.brand ? `<span class="badge">${escapeHtml(seg.brand)}</span>` : ""}
        <span>${seg.start_s.toFixed(1)}s &rarr; ${seg.end_s.toFixed(1)}s</span>
      </div>
      <div class="desc">${escapeHtml(seg.description || "")}</div>
    `;
    segmentList.appendChild(card);
  });
}

function seekTo(seconds) {
  if (!isNaN(player.duration)) {
    player.currentTime = seconds;
    player.play().catch(() => {});
  } else {
    player.currentTime = seconds;
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
