"""
Procedural embroidery artwork: Sisyphus pushing the stone up (left back side)
and the stone rolling down (right back side). A mountain peak sits on the
center back seam.

Canvas is 2 x 1.5 design units (W x H). x=0 is the wearer's LEFT side when
seen from behind (viewer's left), x=2 is the wearer's right.

Run standalone inside Blender to regenerate the PNG:
  Blender -b --python scripts/embroidery_art.py
"""
import os
import numpy as np

# ---- thread palette (linear-ish RGB, saved as sRGB PNG) ----
ORANGE = (0.90, 0.52, 0.14)   # Levi's-style topstitch orange
CREAM = (0.95, 0.90, 0.78)
DARK = (0.16, 0.10, 0.07)
STONE = (0.62, 0.60, 0.56)
STONE_DARK = (0.38, 0.36, 0.33)

# ---- composition parameters ----
CANVAS_H = 1.5        # canvas is 2 x CANVAS_H design units
GROUND_Y0 = 0.10      # ridge height at the outer edges
EDGE_X = 0.04         # ridge starts/ends this far from the canvas sides
PEAK = (1.0, 1.30)    # mountain peak on the center seam
STONE_R = 0.12
FIGURE_SCALE = 1.5    # size of Sisyphus + stones relative to the original sketch

# the figures were drawn on a 31 deg slope; they get rotated onto the real one
_SKETCH_K = 0.61
_SKETCH_PEAK = (1.0, 0.72)


def _grid(w, h):
    xs = (np.arange(w) + 0.5) / w * 2.0
    ys = (np.arange(h) + 0.5) / h * CANVAS_H
    return np.meshgrid(xs, ys)  # rows = y from bottom (Blender pixel order)


class Canvas:
    def __init__(self, w=2048, h=1536):
        self.w, self.h = w, h
        self.X, self.Y = _grid(w, h)
        self.px = 2.0 / w
        self.rgba = np.zeros((h, w, 4), dtype=np.float32)
        self.set_transform()

    def set_transform(self, anchor_from=(0, 0), anchor_to=(0, 0), angle_deg=0.0, scale=1.0):
        """Similarity transform applied to all following primitive coordinates."""
        a = np.radians(angle_deg)
        self._xf = (np.array(anchor_from, float), np.array(anchor_to, float),
                    np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * scale)
        self._s = scale

    def P(self, p):
        f, t, m = self._xf
        return tuple(t + m @ (np.asarray(p, float) - f))

    def paint(self, sdf, color, alpha=1.0):
        cov = np.clip(0.5 - sdf / self.px, 0.0, 1.0) * alpha
        c = self.rgba
        a_old = c[..., 3]
        a_new = cov + a_old * (1 - cov)
        for i in range(3):
            num = color[i] * cov + c[..., i] * a_old * (1 - cov)
            c[..., i] = np.where(a_new > 0, num / np.maximum(a_new, 1e-6), 0)
        c[..., 3] = a_new

    def paint_outlined(self, sdf, fill, outline=DARK, w=0.006):
        self.paint(sdf - w, outline)
        self.paint(sdf, fill)

    # --- signed distance primitives ---
    def circle(self, cx, cy, r):
        cx, cy = self.P((cx, cy))
        return np.hypot(self.X - cx, self.Y - cy) - r * self._s

    def seg(self, a, b, r):
        ax, ay = self.P(a)
        bx, by = self.P(b)
        r *= self._s
        pax, pay = self.X - ax, self.Y - ay
        bax, bay = bx - ax, by - ay
        h = np.clip((pax * bax + pay * bay) / (bax * bax + bay * bay), 0, 1)
        return np.hypot(pax - bax * h, pay - bay * h) - r

    def polyline(self, pts, r):
        d = None
        for a, b in zip(pts[:-1], pts[1:]):
            s = self.seg(a, b, r)
            d = s if d is None else np.minimum(d, s)
        return d

    def arc(self, c, rad, a0, a1, r, n=24):
        t = np.linspace(np.radians(a0), np.radians(a1), n)
        pts = [(c[0] + rad * np.cos(v), c[1] + rad * np.sin(v)) for v in t]
        return self.polyline(pts, r)


K_REAL = (PEAK[1] - GROUND_Y0) / (PEAK[0] - EDGE_X)
SLOPE_DELTA = np.degrees(np.arctan(K_REAL) - np.arctan(_SKETCH_K))


def ground_left(x):
    """Sketch-space ridge (31 deg) used to place the figures."""
    return _SKETCH_PEAK[1] - _SKETCH_K * (_SKETCH_PEAK[0] - x)


def ground_right(x):
    return _SKETCH_PEAK[1] - _SKETCH_K * (x - _SKETCH_PEAK[0])


def real_left(x):
    return GROUND_Y0 + K_REAL * (x - EDGE_X)


def real_right(x):
    return PEAK[1] - K_REAL * (x - PEAK[0])


def along_slope(start, dist, downhill_right):
    u = np.array([1.0, -K_REAL if downhill_right else K_REAL]) / np.hypot(1.0, K_REAL)
    return tuple(np.array(start) + u * dist)


def draw_ridge(cv):
    left, right = (EDGE_X, GROUND_Y0), (2 - EDGE_X, GROUND_Y0)
    cv.set_transform()
    cv.paint(cv.polyline([left, PEAK, right], 0.011), ORANGE)
    # short hatch stitches under the ridge
    for x in np.arange(0.10, 1.95, 0.06):
        if abs(x - PEAK[0]) < 0.04:
            continue
        g = real_left(x) if x < PEAK[0] else real_right(x)
        cv.paint(cv.seg((x, g - 0.03), (x + 0.015, g - 0.065), 0.0035), ORANGE, 0.85)


def draw_sisyphus_pushing(cv):
    """Left side: figure leaning into the stone, pushing it uphill (to the right)."""
    back_foot_sketch = (0.38, ground_left(0.38))
    cv.set_transform(back_foot_sketch, along_slope((EDGE_X, GROUND_Y0), 0.30, False),
                     SLOPE_DELTA, FIGURE_SCALE)
    k = _SKETCH_K
    n = np.array([-k, 1.0]) / np.hypot(k, 1.0)        # slope normal
    contact_x = 0.85
    sc = np.array([contact_x, ground_left(contact_x)]) + n * STONE_R
    stone = cv.circle(sc[0], sc[1], STONE_R)
    cv.paint_outlined(stone, STONE, STONE_DARK, 0.007)
    for a0, a1, rr in ((200, 260, 0.07), (20, 70, 0.08), (110, 150, 0.05)):
        cv.paint(cv.arc(sc, rr, a0, a1, 0.004), STONE_DARK)

    hip, shoulder = (0.50, 0.50), (0.62, 0.62)
    back_knee, back_foot = (0.44, 0.42), (0.38, 0.34)
    front_knee, front_foot = (0.58, 0.56), (0.58, ground_left(0.58))
    d = cv.seg(hip, shoulder, 0.030)
    d = np.minimum(d, cv.seg((0.57, 0.57), (0.615, 0.615), 0.037))   # chest
    d = np.minimum(d, cv.circle(0.648, 0.678, 0.038))                 # bowed head
    d = np.minimum(d, cv.polyline([hip, back_knee, back_foot], 0.019))
    d = np.minimum(d, cv.seg((back_foot[0] - 0.01, back_foot[1] - 0.006),
                             (back_foot[0] + 0.03, back_foot[1] + 0.012), 0.011))
    d = np.minimum(d, cv.polyline([hip, front_knee, front_foot], 0.019))
    d = np.minimum(d, cv.seg((front_foot[0] - 0.01, front_foot[1] + 0.006),
                             (front_foot[0] + 0.035, front_foot[1] + 0.026), 0.011))
    d = np.minimum(d, cv.polyline([shoulder, (0.66, 0.60), (0.70, 0.645)], 0.014))
    d = np.minimum(d, cv.polyline([(0.615, 0.625), (0.645, 0.668), (0.675, 0.69)], 0.014))
    cv.paint_outlined(d, CREAM, DARK, 0.006)


def draw_stone_rolling(cv):
    """Right side: the stone tumbling down the slope with speed lines."""
    k = _SKETCH_K
    t = np.array([1.0, -k]) / np.hypot(1.0, k)          # downhill tangent
    n = np.array([k, 1.0]) / np.hypot(1.0, k)           # slope normal
    contact_x = 1.55
    cv.set_transform((contact_x, ground_right(contact_x)), along_slope(PEAK, 0.95, True),
                     -SLOPE_DELTA, FIGURE_SCALE)
    sc = np.array([contact_x, ground_right(contact_x)]) + n * (STONE_R + 0.035)

    # speed lines trailing uphill
    for off, l0, l1 in ((-0.065, 0.17, 0.36), (0.0, 0.16, 0.46), (0.065, 0.17, 0.33)):
        a = sc - t * l0 + n * off
        b = sc - t * l1 + n * off
        cv.paint(cv.seg(tuple(a), tuple(b), 0.0055), CREAM)
    # dust puffs where it last bounced
    bounce = np.array([1.40, ground_right(1.40)])
    for dx, dy, r in ((0.0, 0.025, 0.022), (-0.04, 0.018, 0.016), (0.035, 0.015, 0.014)):
        c = cv.circle(bounce[0] + dx, bounce[1] + dy, r)
        cv.paint(np.abs(c) - 0.004, CREAM, 0.9)

    stone = cv.circle(sc[0], sc[1], STONE_R)
    cv.paint_outlined(stone, STONE, STONE_DARK, 0.007)
    for a0, a1, rr in ((150, 210, 0.07), (330, 380, 0.08), (60, 100, 0.05)):
        cv.paint(cv.arc(sc, rr, a0, a1, 0.004), STONE_DARK)
    # spin arrow around the stone (clockwise = rolling downhill to the right)
    cv.paint(cv.arc(sc, STONE_R + 0.035, 150, 70, 0.005), ORANGE)
    tip = sc + (STONE_R + 0.035) * np.array([np.cos(np.radians(70)), np.sin(np.radians(70))])
    cv.paint(cv.seg(tuple(tip), (tip[0] - 0.03, tip[1] + 0.012), 0.005), ORANGE)
    cv.paint(cv.seg(tuple(tip), (tip[0] - 0.012, tip[1] + 0.03), 0.005), ORANGE)


def build(w=2048, h=1536):
    cv = Canvas(w, h)
    draw_ridge(cv)
    draw_stone_rolling(cv)
    draw_sisyphus_pushing(cv)
    return cv.rgba


def save_png(path, w=2048, h=1536):
    import bpy
    rgba = build(w, h)
    name = os.path.basename(path)
    img = bpy.data.images.get(name) or bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print("saved", save_png(os.path.join(here, "..", "textures", "embroidery.png")))
