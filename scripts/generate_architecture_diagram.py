"""One-off script to render architecture/option_b_architecture.png. Not part
of the runtime pipeline — run manually if the diagram ever needs updating:
    python scripts/generate_architecture_diagram.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(11, 14))
ax.set_xlim(0, 11)
ax.set_ylim(0, 17)
ax.axis("off")

COLOR_REQUIRED = "#4a7dff"   # our required FastAPI / core pipeline
COLOR_OPTIONAL = "#ff9f43"   # optional internal VLM providers
COLOR_TEST = "#8e8e93"       # test/client tools
COLOR_DATA = "#2ecc71"       # data artifacts / output


def box(x, y, w, h, text, color, fontsize=9, text_color="white"):
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.2, edgecolor="#22242a", facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color, wrap=True, weight="bold")
    return (x + w / 2, y, x + w / 2, y + h)


def arrow(x1, y1, x2, y2, color="#555"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.3, color=color)
    ax.add_patch(a)


# --- Title ---
ax.text(5.5, 16.5, "Option B — Ad Segment Detector Architecture", ha="center", fontsize=15, weight="bold")

# --- Input ---
cx1, cy1_bot, cx1b, cy1_top = box(4, 15.3, 3, 0.6, "INPUT: VIDEO URL / LOCAL FILE", "#333844", 10)

# --- FastAPI (REQUIRED API) ---
cx2, cy2_bot, cx2b, cy2_top = box(3.5, 14.2, 4, 0.7, "FastAPI  —  OUR REQUIRED HTTP API\nPOST /api/v1/detect", COLOR_REQUIRED, 9.5)
arrow(cx1, cy1_bot, cx2, cy2_top)

# --- Acquisition ---
cx3, cy3_bot, cx3b, cy3_top = box(3.5, 13.0, 4, 0.7, "Video Acquisition\n(yt-dlp for YouTube / local file for Reels)", COLOR_REQUIRED, 9)
arrow(cx2, cy2_bot, cx3, cy3_top)

cx4, cy4_bot, cx4b, cy4_top = box(3.5, 11.9, 4, 0.6, "FFmpeg (merge / probe)", COLOR_REQUIRED, 9)
arrow(cx3, cy3_bot, cx4, cy4_top)

cx5, cy5_bot, cx5b, cy5_top = box(4, 10.9, 3, 0.6, "video.mp4 (local)", COLOR_DATA, 9, "#111")
arrow(cx4, cy4_bot, cx5, cy5_top)

# --- Split: Audio vs Frames ---
audio_x, frames_x = 2.2, 6.8
box_w = 3.4

ax_x, ax_bot, ax_top_x, ax_top = box(audio_x - box_w/2, 9.6, box_w, 0.6, "Audio extraction (ffmpeg)", COLOR_REQUIRED, 9)
fr_x, fr_bot, fr_top_x, fr_top = box(frames_x - box_w/2, 9.6, box_w, 0.6, "Frame sampling (configurable + scene-cut aware)", COLOR_REQUIRED, 8.5)
arrow(cx5, cy5_bot, ax_x, ax_top)
arrow(cx5, cy5_bot, fr_x, fr_top)

wh_x, wh_bot, wh_top_x, wh_top = box(audio_x - box_w/2, 8.5, box_w, 0.6, "Whisper ASR (timestamped)", COLOR_REQUIRED, 9)
arrow(ax_x, ax_bot, wh_x, wh_top)

ocr_x, ocr_bot, ocr_top_x, ocr_top = box(frames_x - box_w/2 - 1.8, 8.5, 1.9, 0.6, "OCR\n(EasyOCR)", COLOR_REQUIRED, 8.5)
vlm_x, vlm_bot, vlm_top_x, vlm_top = box(frames_x - box_w/2 + 1.6, 8.5, 1.9, 0.6, "VLM\n(selective calls)", COLOR_REQUIRED, 8.5)
arrow(fr_x, fr_bot, ocr_x, ocr_top)
arrow(fr_x, fr_bot, vlm_x, vlm_top)

# VLM providers (optional, internal)
gem_x, gem_bot, gem_top_x, gem_top = box(6.0, 7.3, 1.6, 0.5, "Gemini", COLOR_OPTIONAL, 8.5)
opr_x, opr_bot, opr_top_x, opr_top = box(7.8, 7.3, 1.9, 0.5, "OpenRouter", COLOR_OPTIONAL, 8.5)
mock_x, mock_bot, mock_top_x, mock_top = box(4.1, 7.3, 1.7, 0.5, "mock/local", COLOR_OPTIONAL, 8.5)
arrow(vlm_x, vlm_bot, gem_x, gem_top)
arrow(vlm_x, vlm_bot, opr_x, opr_top)
arrow(vlm_x, vlm_bot, mock_x, mock_top)
ax.text(9.9, 7.55, "OPTIONAL\ninternal VLM\nproviders\n(configured via\nVLM_PROVIDER,\nNOT separate\nendpoints)",
        fontsize=7.5, color=COLOR_OPTIONAL, ha="left", va="center", weight="bold")

# --- Signal fusion ---
transcript_bottom_y = wh_bot
ocr_bottom_y = ocr_bot
vlm_bottom_y = 7.3

fusion_x, fusion_bot, fusion_top_x, fusion_top = box(3.75, 6.1, 3.5, 0.6, "Signal Fusion\n(weighted ASR + OCR + VLM + scene-cut)", COLOR_REQUIRED, 9)
arrow(wh_x, transcript_bottom_y, fusion_x, fusion_top)
arrow(ocr_x, ocr_bottom_y, fusion_x, fusion_top)
arrow(4.1 + 0.85, vlm_bottom_y, fusion_x, fusion_top)

# --- Detection + segmentation ---
det_x, det_bot, det_top_x, det_top = box(3.75, 5.0, 3.5, 0.6, "Ad Detection + Taxonomy Classification", COLOR_REQUIRED, 9)
arrow(fusion_x, fusion_bot, det_x, det_top)

seg_x, seg_bot, seg_top_x, seg_top = box(3.75, 3.9, 3.5, 0.6, "Temporal Segmentation (merge / split rules)", COLOR_REQUIRED, 9)
arrow(det_x, det_bot, seg_x, seg_top)

json_x, json_bot, json_top_x, json_top = box(3.5, 2.8, 4, 0.6, "REQUIRED JSON CONTRACT\n(source / segments / evidence / stats)", COLOR_DATA, 9, "#111")
arrow(seg_x, seg_bot, json_x, json_top)

# --- Consumers ---
view_x, view_bot, view_top_x, view_top = box(1.5, 1.6, 2.6, 0.6, "Minimal HTML Viewer", COLOR_REQUIRED, 9)
eval_x, eval_bot, eval_top_x, eval_top = box(6.9, 1.6, 2.6, 0.6, "Evaluation (IoU, boundary error)", COLOR_REQUIRED, 8.5)
arrow(json_x, json_bot, view_x, view_top)
arrow(json_x, json_bot, eval_x, eval_top)

metrics_x, metrics_bot, metrics_top_x, metrics_top = box(6.9, 0.5, 2.6, 0.6, "Metrics + Failure Analysis", COLOR_DATA, 8.5, "#111")
arrow(eval_x, eval_bot, metrics_x, metrics_top)

# --- Legend ---
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_REQUIRED, markersize=14, label="Our required pipeline / FastAPI"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_OPTIONAL, markersize=14, label="Optional internal VLM provider"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_DATA, markersize=14, label="Data artifact / output"),
]
ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.0, -0.02), fontsize=8.5, frameon=False)

ax.text(
    5.5, -0.15,
    "Note: Swagger UI (/docs), curl, and Postman are only test clients for the FastAPI service above —\n"
    "they are not part of the architecture and are not to be confused with the optional VLM providers.",
    ha="center", fontsize=8, color="#666", style="italic",
)

plt.tight_layout()
plt.savefig("architecture/option_b_architecture.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved architecture/option_b_architecture.png")
