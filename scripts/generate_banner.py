#!/usr/bin/env python3
"""Generate dark.svg / light.svg GitHub profile banners for prerak2612."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "source-photo.png"
OUT_DARK = ROOT / "dark.svg"
OUT_LIGHT = ROOT / "light.svg"

W, H = 1180, 610
PORTRAIT_COLS, PORTRAIT_ROWS = 300, 340
PANEL_X, PANEL_Y = 48, 78
PANEL_W, PANEL_H = 420, 470  # ~38% visual area
INFO_X = 510

# Palette
BG = "#0A101F"
PORTRAIT_DARK = "#A78BFA"
PORTRAIT_LIGHT = "#7C3AED"
CHROME_DARK = "#22D3EE"
CHROME_LIGHT = "#0891B2"
ACCENT = "#10B981"
MUTED = "#94A3B8"
TEXT = "#F8FAFC"
LIVE_RED = "#EF4444"

PROFILE = {
    "name": "Prerak Arya",
    "handle": "prerak2612",
    "subject": "Prerak Arya",
    "origin": "India",
    "education": "B.Tech CSE (AI)",
    "status": "Building + Learning + Shipping",
    "toolchain": "VS Code · Git · Postman",
    "lang": "JS · TypeScript",
    "frontend": "React · Next.js · Tailwind",
    "backend": "Node.js",
    "database": "SQLite",
    "infra": "GitHub · Vercel · Netlify",
    "mail": "prerakarya2612@gmail.com",
    "portfolio": "personal-portfolio-peach-one-50.vercel.app",
    "linkedin": "linkedin.com/in/prerak-arya-a60b89269",
    "github": "prerak2612",
    "facebook": "—",
}


@dataclass
class Dot:
    x: float
    y: float
    w: float = 1.0


def load_and_prep_photo() -> tuple[Image.Image, np.ndarray]:
    im = Image.open(PHOTO).convert("RGB")
    arr = np.asarray(im).astype(np.float32)
    # Circular portrait sits on dark slate bg — segment subject
    bg = arr[10, 10]
    dist = np.linalg.norm(arr - bg, axis=2)
    mask = dist > 28
    # Keep largest component near center
    labeled, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        mask = labeled == keep
        mask = ndimage.binary_closing(mask, iterations=3)
        mask = ndimage.binary_fill_holes(mask)
    # Crop to subject bbox with head-shoulders padding
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad_x = int((x1 - x0) * 0.08)
    pad_top = int((y1 - y0) * 0.06)
    pad_bot = int((y1 - y0) * 0.12)
    x0 = max(0, x0 - pad_x)
    x1 = min(im.width - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_top)
    y1 = min(im.height - 1, y1 + pad_bot)
    crop = im.crop((x0, y0, x1 + 1, y1 + 1))
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    # Fit into portrait frame aspect (~300:340)
    target_aspect = PORTRAIT_COLS / PORTRAIT_ROWS
    cw, ch = crop.size
    cur = cw / ch
    if cur > target_aspect:
        nw = int(ch * target_aspect)
        left = (cw - nw) // 2
        crop = crop.crop((left, 0, left + nw, ch))
        crop_mask = crop_mask[:, left : left + nw]
    else:
        nh = int(cw / target_aspect)
        top = max(0, (ch - nh) // 5)  # bias upward for face
        if top + nh > ch:
            top = ch - nh
        crop = crop.crop((0, top, cw, top + nh))
        crop_mask = crop_mask[top : top + nh, :]
    crop = crop.resize((PORTRAIT_COLS, PORTRAIT_ROWS), Image.Resampling.LANCZOS)
    crop_mask = np.array(
        Image.fromarray(crop_mask.astype(np.uint8) * 255).resize(
            (PORTRAIT_COLS, PORTRAIT_ROWS), Image.Resampling.NEAREST
        )
    ) > 127
    # Contrast pipeline
    crop = ImageOps.autocontrast(crop, cutoff=1)
    crop = ImageEnhance.Contrast(crop).enhance(1.3)
    crop = crop.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=2))
    return crop, crop_mask


def floyd_steinberg(gray: np.ndarray, serpentine: bool = True) -> np.ndarray:
    """1-bit Floyd–Steinberg dither. Returns boolean True where ink/dot."""
    g = gray.astype(np.float64).copy()
    h, w = g.shape
    out = np.zeros((h, w), dtype=bool)
    for y in range(h):
        xs = range(w) if (not serpentine or y % 2 == 0) else range(w - 1, -1, -1)
        for x in xs:
            old = g[y, x]
            new = 0.0 if old < 128 else 255.0
            out[y, x] = new == 0.0  # ink on dark parts
            err = old - new
            if serpentine and y % 2:
                # leftward
                if x - 1 >= 0:
                    g[y, x - 1] += err * 7 / 16
                if y + 1 < h:
                    if x + 1 < w:
                        g[y + 1, x + 1] += err * 3 / 16
                    g[y + 1, x] += err * 5 / 16
                    if x - 1 >= 0:
                        g[y + 1, x - 1] += err * 1 / 16
            else:
                if x + 1 < w:
                    g[y, x + 1] += err * 7 / 16
                if y + 1 < h:
                    if x - 1 >= 0:
                        g[y + 1, x - 1] += err * 3 / 16
                    g[y + 1, x] += err * 5 / 16
                    if x + 1 < w:
                        g[y + 1, x + 1] += err * 1 / 16
    return out


def portrait_dots(mode: str, target: int = 17000) -> list[Dot]:
    crop, mask = load_and_prep_photo()
    gray = np.asarray(ImageOps.grayscale(crop)).astype(np.float64)
    if mode == "dark":
        # dots draw lit subject on dark panel
        bright = floyd_steinberg(255 - gray)
        ink = bright & mask
        ink &= mask
    else:
        # light mode: dark ink of subject; drop empty circular backdrop
        ink = floyd_steinberg(gray) & mask
    ys, xs = np.where(ink)
    # Map to panel coords (leave margin inside VISUAL.MAP frame)
    pad = 18
    inner_w = PANEL_W - 2 * pad
    inner_h = PANEL_H - 48 - pad  # leave room for label
    origin_x = PANEL_X + pad
    origin_y = PANEL_Y + 36
    coords = np.stack([xs, ys], axis=1)
    if len(coords) > target:
        rng = np.random.default_rng(0)
        # Prefer denser facial region (upper-mid) slightly via weighted sample
        weights = 1.0 + 0.35 * (1.0 - np.abs((coords[:, 1] / PORTRAIT_ROWS) - 0.38))
        weights = weights / weights.sum()
        pick = rng.choice(len(coords), size=target, replace=False, p=weights)
        coords = coords[pick]
    dots: list[Dot] = []
    for x, y in coords:
        px = origin_x + (x + 0.5) / PORTRAIT_COLS * inner_w
        py = origin_y + (y + 0.5) / PORTRAIT_ROWS * inner_h
        dots.append(Dot(float(px), float(py)))
    return dots


def path_runs(dots: list[Dot], color: str, opacity: float = 1.0, gid: str = "") -> str:
    """Render dots as horizontal path runs with crispEdges."""
    by_y: dict[float, list[float]] = defaultdict(list)
    for d in dots:
        by_y[round(d.y, 2)].append(d.x)
    parts = []
    for y, xs in sorted(by_y.items()):
        xs = sorted(xs)
        run_start = xs[0]
        prev = xs[0]
        for x in xs[1:]:
            if x - prev <= 1.35:
                prev = x
                continue
            # flush
            parts.append(f"M{run_start:.2f},{y:.2f}h{(prev - run_start) + 0.9:.2f}")
            run_start = x
            prev = x
        parts.append(f"M{run_start:.2f},{y:.2f}h{(prev - run_start) + 0.9:.2f}")
    d = "".join(parts)
    op = f' opacity="{opacity:.3f}"' if opacity < 1 else ""
    gid_attr = f' id="{gid}"' if gid else ""
    return (
        f'<path{gid_attr} d="{d}" stroke="{color}" stroke-width="0.95" '
        f'fill="none" stroke-linecap="square" shape-rendering="crispEdges"{op}/>'
    )


def logo_point_clouds(n: int = 900) -> dict[str, np.ndarray]:
    """Three logo silhouettes as Nx2 point clouds in portrait panel space."""
    pad = 18
    ox = PANEL_X + pad
    oy = PANEL_Y + 36
    iw = PANEL_W - 2 * pad
    ih = PANEL_H - 48 - pad
    cx, cy = ox + iw / 2, oy + ih / 2

    def sample_mask(mask: np.ndarray, count: int) -> np.ndarray:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return np.zeros((count, 2))
        idx = np.random.choice(len(xs), size=count, replace=True)
        # normalize mask coords to panel
        h, w = mask.shape
        pts = np.stack(
            [
                ox + (xs[idx] + 0.5) / w * iw,
                oy + (ys[idx] + 0.5) / h * ih,
            ],
            axis=1,
        )
        return pts

    rng = np.random.default_rng(42)
    # React atom
    m = np.zeros((200, 200), dtype=bool)
    for ang in (0, 60, 120):
        a = math.radians(ang)
        for t in np.linspace(0, 2 * math.pi, 400):
            # ellipse orbit
            ex = 70 * math.cos(t)
            ey = 28 * math.sin(t)
            x = 100 + ex * math.cos(a) - ey * math.sin(a)
            y = 100 + ex * math.sin(a) + ey * math.cos(a)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    xi, yi = int(x + dx), int(y + dy)
                    if 0 <= xi < 200 and 0 <= yi < 200:
                        m[yi, xi] = True
    # nucleus
    yy, xx = np.ogrid[:200, :200]
    m |= (xx - 100) ** 2 + (yy - 100) ** 2 <= 10**2
    react = sample_mask(m, n)

    # </> code glyph
    m = np.zeros((200, 200), dtype=bool)
    # left <
    for i, y in enumerate(np.linspace(40, 100, 80)):
        x = 90 - (y - 40) * 0.7
        for t in range(-3, 4):
            m[int(y) + t // 2, int(x) + t] = True
    for i, y in enumerate(np.linspace(100, 160, 80)):
        x = 90 - (160 - y) * 0.7
        for t in range(-3, 4):
            m[int(y) + t // 2, int(x) + t] = True
    # right >
    for y in np.linspace(40, 100, 80):
        x = 110 + (y - 40) * 0.7
        for t in range(-3, 4):
            m[int(y) + t // 2, int(x) + t] = True
    for y in np.linspace(100, 160, 80):
        x = 110 + (160 - y) * 0.7
        for t in range(-3, 4):
            m[int(y) + t // 2, int(x) + t] = True
    # slash
    for i in range(90):
        x = 115 - i * 0.35
        y = 45 + i * 1.2
        for t in range(-2, 3):
            xi, yi = int(x + t), int(y)
            if 0 <= xi < 200 and 0 <= yi < 200:
                m[yi, xi] = True
    code = sample_mask(m, n)

    # Next.js N mark (stylized)
    m = np.zeros((200, 200), dtype=bool)
    # left bar
    m[40:160, 55:70] = True
    # right bar
    m[40:160, 130:145] = True
    # diagonal
    for i in range(120):
        x = 70 + i * 0.55
        y = 40 + i
        for t in range(-4, 5):
            xi, yi = int(x + t), int(y)
            if 0 <= xi < 200 and 0 <= yi < 200:
                m[yi, xi] = True
    # circle outline
    for ang in np.linspace(0, 2 * math.pi, 360):
        x = 100 + 78 * math.cos(ang)
        y = 100 + 78 * math.sin(ang)
        for t in range(-2, 3):
            xi, yi = int(x + t), int(y + t // 2)
            if 0 <= xi < 200 and 0 <= yi < 200:
                m[yi, xi] = True
    nxt = sample_mask(m, n)
    return {"react": react, "code": code, "next": nxt}


def match_transport(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Greedy subsampled assignment for morph paths. Returns permuted dst."""
    n = len(src)
    # subsample for cost matrix if large
    if n > 400:
        # block matching via nearest after sorting by angle/radius
        sc = src - src.mean(0)
        dc = dst - dst.mean(0)
        sa = np.arctan2(sc[:, 1], sc[:, 0])
        da = np.arctan2(dc[:, 1], dc[:, 0])
        order_s = np.argsort(sa)
        order_d = np.argsort(da)
        out = np.zeros_like(dst)
        out[order_s] = dst[order_d]
        # local refine with noise jitter already in logos
        return out
    cost = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)
    _, col = linear_sum_assignment(cost)
    return dst[col]


def group_intro(dots: list[Dot], groups: int = 60) -> list[list[Dot]]:
    idx = list(range(len(dots)))
    random.Random(7).shuffle(idx)
    buckets = [[] for _ in range(groups)]
    for i, di in enumerate(idx):
        buckets[i % groups].append(dots[di])
    return buckets


def drift_bands(dots: list[Dot], bands: int = 94, target: tuple[float, float] | None = None) -> list[list[Dot]]:
    rng = np.random.default_rng(11)
    keyed = []
    for d in dots:
        # noise before grouping to avoid grid dissolve
        key = d.y + float(rng.normal(0, 4))
        keyed.append((key, d))
    keyed.sort(key=lambda t: t[0])
    size = max(1, len(keyed) // bands)
    out = []
    for i in range(0, len(keyed), size):
        out.append([d for _, d in keyed[i : i + size]])
    return out[:bands] if len(out) > bands else out


def text_row(y: float, label: str, value: str, color: str, label_w: float = 118) -> str:
    """Row with dotted leader and locked textLength alignment."""
    x0 = INFO_X
    value_x = 1120
    # approximate char widths
    label_len = max(1, len(label))
    value_len = max(1, len(value))
    label_tl = min(label_w, 7.2 * label_len)
    value_tl = min(220, 7.0 * value_len)
    # dots between
    gap_start = x0 + label_tl + 8
    gap_end = value_x - value_tl - 8
    dots = ""
    if gap_end > gap_start:
        # svg dotted line
        dots = (
            f'<line x1="{gap_start:.1f}" y1="{y}" x2="{gap_end:.1f}" y2="{y}" '
            f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="1 4" opacity="0.55"/>'
        )
    return (
        f'<text x="{x0}" y="{y}" fill="{MUTED}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
        f'font-size="14" textLength="{label_tl:.1f}" lengthAdjust="spacingAndGlyphs">{label}</text>'
        f"{dots}"
        f'<text x="{value_x}" y="{y}" fill="{color}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
        f'font-size="14" text-anchor="end" textLength="{value_tl:.1f}" lengthAdjust="spacingAndGlyphs">{value}</text>'
    )


def chrome(mode: str) -> str:
    chrome = CHROME_DARK if mode == "dark" else CHROME_LIGHT
    bg = BG if mode == "dark" else "#F8FAFC"
    panel_fill = "#070B14" if mode == "dark" else "#FFFFFF"
    border = "#1E293B" if mode == "dark" else "#CBD5E1"
    title_fg = TEXT if mode == "dark" else "#0F172A"
    return f'''
  <rect width="{W}" height="{H}" rx="18" fill="{bg}"/>
  <rect x="20" y="20" width="1140" height="570" rx="14" fill="{panel_fill}" stroke="{border}" stroke-width="1.5"/>
  <!-- title bar -->
  <rect x="20" y="20" width="1140" height="40" rx="14" fill="{border}" opacity="0.35"/>
  <rect x="20" y="40" width="1140" height="20" fill="{panel_fill}"/>
  <circle cx="48" cy="40" r="6" fill="#FF5F56"/>
  <circle cx="70" cy="40" r="6" fill="#FFBD2E"/>
  <circle cx="92" cy="40" r="6" fill="#27C93F"/>
  <text x="590" y="45" text-anchor="middle" fill="{chrome}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="13">profile.sh --live</text>
  <!-- visual map frame -->
  <rect x="{PANEL_X}" y="{PANEL_Y}" width="{PANEL_W}" height="{PANEL_H}" rx="10" fill="none" stroke="{chrome}" stroke-width="1.4"/>
  <text x="{PANEL_X + 14}" y="{PANEL_Y + 22}" fill="{chrome}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="12" letter-spacing="1.5">VISUAL.MAP</text>
  <!-- LIVE badge -->
  <g transform="translate(980, 92)">
    <rect x="0" y="-14" width="58" height="20" rx="4" fill="{LIVE_RED}" opacity="0.15"/>
    <circle cx="10" cy="-4" r="4" fill="{LIVE_RED}">
      <animate attributeName="opacity" values="1;0.25;1" dur="1.6s" repeatCount="indefinite"/>
    </circle>
    <text x="20" y="0" fill="{LIVE_RED}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="12" font-weight="700">LIVE</text>
  </g>
  <!-- handle pill -->
  <g transform="translate(860, 118)">
    <rect x="0" y="-16" width="178" height="28" rx="14" fill="{chrome}" opacity="0.12" stroke="{chrome}" stroke-width="1"/>
    <text x="89" y="2" text-anchor="middle" fill="{chrome}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="14">@{PROFILE["handle"]}</text>
  </g>
  <text x="{INFO_X}" y="88" fill="{chrome}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="13" letter-spacing="1.5">SYSTEM.INFO</text>
'''


def info_panel(mode: str) -> str:
    color = TEXT if mode == "dark" else "#0F172A"
    rows = [
        ("Subject", PROFILE["subject"]),
        ("Origin", PROFILE["origin"]),
        ("Education", PROFILE["education"]),
        ("Status", PROFILE["status"]),
        ("ToolChain", PROFILE["toolchain"]),
        ("Core.Lang", PROFILE["lang"]),
        ("Core.Frontend", PROFILE["frontend"]),
        ("Core.Backend", PROFILE["backend"]),
        ("Core.Database", PROFILE["database"]),
        ("Core.Infra", PROFILE["infra"]),
        ("Grid.Mail", PROFILE["mail"]),
        ("Grid.Portfolio", PROFILE["portfolio"]),
        ("Grid.LinkedIn", PROFILE["linkedin"]),
        ("Grid.GitHub", PROFILE["github"]),
    ]
    y0 = 150
    spacing = 23
    parts = [text_row(y0 + i * spacing, lab, val, color) for i, (lab, val) in enumerate(rows)]
    return "\n".join(parts)


def build_svg(mode: str) -> str:
    random.seed(42)
    np.random.seed(42)
    portrait_color = PORTRAIT_DARK if mode == "dark" else PORTRAIT_LIGHT
    chrome_c = CHROME_DARK if mode == "dark" else CHROME_LIGHT
    dots = portrait_dots(mode)
    print(f"[{mode}] portrait dots: {len(dots)}")

    # Intro groups
    groups = group_intro(dots, 60)
    intro_layers = []
    for i, g in enumerate(groups):
        delay = (i / 60) * 2.0
        path = path_runs(g, portrait_color)
        intro_layers.append(
            f'<g opacity="0">{path}'
            f'<animate attributeName="opacity" from="0" to="1" begin="{delay:.3f}s" dur="0.35s" fill="freeze"/>'
            f"</g>"
        )

    # Duplicate full portrait for loop (after intro)
    bands = drift_bands(dots, 94)
    # target centroid = first logo (react)
    logos = logo_point_clouds(900)
    react_c = logos["react"].mean(axis=0)

    loop_portrait = []
    # Loop timing: portrait 3.0s, each logo 2.0s, 1.3s transitions — total ~14.2s
    # phases: portrait hold, dissolve to logo1, logo1, to logo2, logo2, to logo3, logo3, back to portrait
    # Use uneven keyTimes on portrait bands
    for bi, band in enumerate(bands):
        # drift ~42% toward react centroid
        dx = (react_c[0] - np.mean([d.x for d in band])) * 0.42
        dy = (react_c[1] - np.mean([d.y for d in band])) * 0.42
        path = path_runs(band, portrait_color)
        # opacity/transform over loop
        loop_portrait.append(
            f'''<g transform="translate(0,0)">
  {path}
  <animateTransform attributeName="transform" type="translate"
    values="0,0;0,0;{dx:.2f},{dy:.2f};{dx:.2f},{dy:.2f};0,0;0,0"
    keyTimes="0;0.21;0.30;0.86;0.95;1"
    dur="14.2s" begin="3.2s" repeatCount="indefinite" calcMode="linear"/>
  <animate attributeName="opacity"
    values="1;1;0;0;0;0;1"
    keyTimes="0;0.21;0.30;0.44;0.86;0.95;1"
    dur="14.2s" begin="3.2s" repeatCount="indefinite"/>
</g>'''
        )

    # Travellers — morph between logos
    # pick subset of portrait dots as travellers start positions
    trav_src = np.array([[d.x, d.y] for d in random.sample(dots, min(900, len(dots)))])
    if len(trav_src) < 900:
        extra = 900 - len(trav_src)
        trav_src = np.vstack([trav_src, logos["react"][:extra]])
    r = logos["react"]
    c = match_transport(r, logos["code"])
    n = match_transport(c, logos["next"])
    # chain: portrait positions -> react -> code -> next -> portrait
    p0 = trav_src
    p1 = match_transport(p0, r)
    p2 = match_transport(p1, c)
    p3 = match_transport(p2, n)
    p4 = match_transport(p3, p0)

    trav_paths = []
    # render as individual small dots for morph (thicker)
    # To keep file smaller, group into runs per frame is hard; use animate motion on batches
    # Compact approach: 30 traveler subgroups
    batch = 30
    for i in range(0, 900, batch):
        sl = slice(i, i + batch)
        # create a path group that interpolates via animate on each point — expensive
        # Instead animate a single path morph using values of d — still heavy
        # Practical: animateTransform between centroids + opacity, and swap path d via discrete values
        frames = [p0[sl], p1[sl], p2[sl], p3[sl], p4[sl]]
        frame_paths = []
        for fr in frames:
            dots_f = [Dot(float(x), float(y), 1.4) for x, y in fr]
            # thicker stroke for travellers
            by_y: dict[float, list[float]] = defaultdict(list)
            for d in dots_f:
                by_y[round(d.y, 1)].append(d.x)
            parts = []
            for y, xs in sorted(by_y.items()):
                for x in xs:
                    parts.append(f"M{x:.1f},{y:.1f}h1.3")
            frame_paths.append("".join(parts))
        # discrete morph of d
        values = ";".join(frame_paths)
        # keyTimes for loop: hidden during portrait (0-0.21), then logos
        trav_paths.append(
            f'''<path d="{frame_paths[0]}" stroke="{chrome_c}" stroke-width="1.35" fill="none"
  stroke-linecap="square" shape-rendering="crispEdges" opacity="0">
  <animate attributeName="d" values="{values}"
    keyTimes="0;0.21;0.44;0.65;1" dur="14.2s" begin="3.2s" repeatCount="indefinite"
    calcMode="linear"/>
  <animate attributeName="opacity" values="0;0;0;1;1;1;1;0"
    keyTimes="0;0.18;0.21;0.30;0.44;0.65;0.90;1" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>
</path>'''
        )

    # After intro finishes, show looping portrait (intro freezes at opacity 1; loop group starts at 3.2s)
    # Hide intro after loop starts to avoid double-drawing
    intro_wrap = (
        '<g id="intro">\n'
        + "\n".join(intro_layers)
        + '\n<animate attributeName="opacity" from="1" to="0" begin="3.2s" dur="0.01s" fill="freeze"/>\n</g>'
    )
    loop_wrap = (
        '<g id="loop" opacity="0">\n'
        + "\n".join(loop_portrait)
        + "\n"
        + "\n".join(trav_paths)
        + '\n<animate attributeName="opacity" from="0" to="1" begin="3.2s" dur="0.01s" fill="freeze"/>\n</g>'
    )

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{PROFILE["name"]} — profile.sh --live">
  <title>{PROFILE["name"]} · profile.sh --live</title>
{chrome(mode)}
{info_panel(mode)}
{intro_wrap}
{loop_wrap}
</svg>
'''
    return svg


def main() -> None:
    for mode, path in (("dark", OUT_DARK), ("light", OUT_LIGHT)):
        svg = build_svg(mode)
        path.write_text(svg)
        kb = path.stat().st_size / 1024
        print(f"Wrote {path.name} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
