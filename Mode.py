import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn

# ----------------------------
# CNN model (same as your MNIST trainer)
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------------------
# Load trained MNIST model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.eval()

# ----------------------------
# GUI setup
# ----------------------------
WIDTH, HEIGHT = 320, 320  # drawing canvas
BG = 255  # white
INK = 0   # black

root = tk.Tk()
root.title("MNIST Multi-Digit Recognizer")

# Layout
top = ttk.Frame(root)
top.pack(padx=10, pady=10)

canvas = tk.Canvas(top, width=WIDTH, height=HEIGHT, bg="white", highlightthickness=1, highlightbackground="#ccc")
canvas.grid(row=0, column=0, rowspan=6)

controls = ttk.Frame(top)
controls.grid(row=0, column=1, sticky="nw", padx=(10,0))

# PIL image to mirror the canvas (for clean pixels)
img = Image.new("L", (WIDTH, HEIGHT), color=BG)
draw = ImageDraw.Draw(img)

# Brush / eraser
brush_size = tk.IntVar(value=18)
eraser_on = tk.BooleanVar(value=False)

def on_paint(event):
    r = brush_size.get() // 2
    x1, y1, x2, y2 = event.x - r, event.y - r, event.x + r, event.y + r
    fill = "white" if eraser_on.get() else "black"
    canvas.create_oval(x1, y1, x2, y2, fill=fill, outline=fill)
    draw.ellipse([x1, y1, x2, y2], fill=(BG if eraser_on.get() else INK))

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0,0,WIDTH,HEIGHT], fill=BG)
    pred_var.set("—")
    probs_var.set("")

def toggle_eraser():
    eraser_on.set(not eraser_on.get())
    eraser_btn.config(text=("Eraser: ON" if eraser_on.get() else "Eraser: OFF"))

# Keyboard shortcuts
root.bind("<e>", lambda e: toggle_eraser())
root.bind("<E>", lambda e: toggle_eraser())
root.bind("<c>", lambda e: clear_canvas())
root.bind("<C>", lambda e: clear_canvas())

canvas.bind("<B1-Motion>", on_paint)
canvas.bind("<Button-1>", on_paint)

# Controls UI
ttk.Label(controls, text="Brush size").grid(row=0, column=0, sticky="w")
ttk.Scale(controls, from_=6, to=36, orient="horizontal", variable=brush_size, length=140).grid(row=1, column=0, pady=(0,8))
eraser_btn = ttk.Button(controls, text="Eraser: OFF", command=toggle_eraser)
eraser_btn.grid(row=2, column=0, sticky="ew")

ttk.Button(controls, text="Clear (C)", command=clear_canvas).grid(row=3, column=0, pady=(8,0), sticky="ew")

ttk.Separator(controls, orient="horizontal").grid(row=4, column=0, sticky="ew", pady=8)
ttk.Label(controls, text="Live Prediction").grid(row=5, column=0, sticky="w")

pred_var = tk.StringVar(value="—")
probs_var = tk.StringVar(value="")
pred_label = ttk.Label(controls, textvariable=pred_var, font=("Segoe UI", 16, "bold"))
pred_label.grid(row=6, column=0, sticky="w", pady=(2,0))
probs_label = ttk.Label(controls, textvariable=probs_var, font=("Consolas", 9), wraplength=180, justify="left")
probs_label.grid(row=7, column=0, sticky="w")

# ----------------------------
# Helpers: segmentation + preprocessing
# ----------------------------
def crop_and_center_28x28(pil_img, padding=4):
    """
    Take a grayscale PIL image of a digit (black on white), crop tight, pad,
    then resize/paste into a 28x28 image preserving aspect, centered.
    """
    arr = np.array(pil_img)
    # find ink
    ink = arr < 200
    if ink.sum() == 0:
        return Image.new("L", (28,28), color=BG)
    ys, xs = np.where(ink)
    x1, x2 = xs.min(), xs.max()+1
    y1, y2 = ys.min(), ys.max()+1
    cropped = pil_img.crop((x1, y1, x2, y2))
    # add padding
    w, h = cropped.size
    padded = Image.new("L", (w+2*padding, h+2*padding), color=BG)
    padded.paste(cropped, (padding, padding))
    # keep aspect: scale longer side to 20, then center in 28x28 (like MNIST style)
    target = Image.new("L", (28,28), color=BG)
    pw, ph = padded.size
    scale = 20 / max(pw, ph)
    nw, nh = max(1, int(pw*scale)), max(1, int(ph*scale))
    small = padded.resize((nw, nh), Image.LANCZOS)
    ox, oy = (28 - nw)//2, (28 - nh)//2
    target.paste(small, (ox, oy))
    return target

def segment_digits(pil_img, min_stroke=10, min_width=14, merge_gap=6, max_digits=8):
    """
    Split a canvas into digit regions by column-projection.
    - min_stroke: minimum total ink to consider non-empty canvas
    - min_width: ignore tiny noise segments smaller than this (in pixels)
    - merge_gap: merge close segments separated by small whitespace
    """
    # ensure black ink on white
    gray = pil_img.convert("L")
    arr = np.array(gray)
    inv = 255 - arr  # ink bright
    # binarize
    thresh = (inv > 40).astype(np.uint8)  # forgiving threshold
    if thresh.sum() < min_stroke:
        return []  # empty

    col_sums = thresh.sum(axis=0)
    cols = len(col_sums)

    # Find spans where columns contain ink
    segments = []
    in_run = False
    start = 0
    for i in range(cols):
        if col_sums[i] > 0 and not in_run:
            in_run = True
            start = i
        elif col_sums[i] == 0 and in_run:
            in_run = False
            end = i
            if end - start >= min_width:
                segments.append([start, end])
    if in_run:
        end = cols
        if end - start >= min_width:
            segments.append([start, end])

    # Merge small gaps
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            prev = merged[-1]
            if seg[0] - prev[1] <= merge_gap:
                prev[1] = seg[1]
            else:
                merged.append(seg)

    # Cut ROIs and return cropped PILs
    rois = []
    for (x1, x2) in merged[:max_digits]:
        roi = gray.crop((x1, 0, x2, gray.height))
        rois.append(roi)
    return rois

def predict_digits_from_canvas():
    # Work on a copy to avoid race with painter
    pil = img.copy()
    # Ensure digits are dark on light
    pil = ImageOps.invert(pil)  # now ink is white-ish
    pil = ImageOps.invert(pil)  # (noop if already black-on-white) keeps consistency

    # Segment digits left→right
    parts = segment_digits(img)

    if not parts:
        pred_var.set("—")
        probs_var.set("")
        return

    preds = []
    probs_all = []

    tensors = []
    for p in parts:
        norm = crop_and_center_28x28(p)              # 28x28 grayscale
        n = np.array(norm).astype(np.float32) / 255.0
        # normalize like training
        n = (n - 0.5) / 0.5
        t = torch.from_numpy(n).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
        tensors.append(t)

    batch = torch.cat(tensors, dim=0).to(device)

    with torch.no_grad():
        out = model(batch)
        pr = torch.softmax(out, dim=1).cpu().numpy()
        cls = pr.argmax(axis=1).tolist()

    preds = cls
    probs_all = pr

    text_digits = "".join(str(d) for d in preds)
    pred_var.set(f"{text_digits}")

    # Show compact probs per digit (top-3 each)
    tops = []
    for i, pr in enumerate(probs_all):
        top3 = np.argsort(-pr)[:3]
        tops.append(f"[{i}:{top3[0]} {pr[top3[0]]:.2f}, {top3[1]} {pr[top3[1]]:.2f}, {top3[2]} {pr[top3[2]]:.2f}]")
    probs_var.set("  ".join(tops))

# Live update
def tick():
    predict_digits_from_canvas()
    root.after(500, tick)  # 0.5s

tick()

root.mainloop()
