"""
Levi's 501-style back details painted in real-world meters (see
assets/reference/levis_501_back.jpg): yoke, center seam, back pockets with
arcuate stitching, belt loops, leather patch, red tab.

Produces two textures over the back of the pants (planar projection, x/z):
  textures/trims_color.png   RGBA (alpha = where the trim covers the denim)
  textures/trims_height.png  grayscale height for bump (0.5 = flat denim)

Coordinates: x in meters (+x = wearer's LEFT = viewer's left from behind), z = height.
"""
import os
import numpy as np

X_HALF = 0.24             # canvas spans x in [-X_HALF, X_HALF]
Z0, Z1 = 0.74, 1.06       # canvas spans these heights
RES = (3072, 2048)        # px (w, h) -> ~0.16 mm per px

THREAD = (0.86, 0.70, 0.40)       # golden topstitch thread
LEATHER = (0.50, 0.34, 0.21)
LEATHER_INK = (0.36, 0.12, 0.08)
RED_TAB = (0.62, 0.04, 0.05)
EDGE_WEAR = (0.30, 0.40, 0.58)    # abraded indigo on raised edges

STITCH_W = 0.00055        # thread half-width
DASH, DUTY = 0.0034, 0.72  # stitch length / fraction that is thread


class Canvas:
    def __init__(self, w, h):
        xs = X_HALF - (np.arange(w) + 0.5) / w * 2 * X_HALF   # image left = +x
        zs = Z0 + (np.arange(h) + 0.5) / h * (Z1 - Z0)
        self.X, self.Z = np.meshgrid(xs, zs)
        self.px = 2 * X_HALF / w
        self.rgba = np.zeros((h, w, 4), np.float32)
        self.height = np.full((h, w), 0.5, np.float32)

    def cov(self, sdf):
        return np.clip(0.5 - sdf / self.px, 0, 1)

    def paint(self, sdf, color, alpha=1.0):
        c = self.cov(sdf) * alpha
        a0 = self.rgba[..., 3]
        a1 = c + a0 * (1 - c)
        for i in range(3):
            num = color[i] * c + self.rgba[..., i] * a0 * (1 - c)
            self.rgba[..., i] = np.where(a1 > 0, num / np.maximum(a1, 1e-6), 0)
        self.rgba[..., 3] = a1

    def lift(self, sdf, h, soft=0.0):
        """Set height to h inside the shape (soft = bevel width in meters)."""
        c = np.clip(0.5 - sdf / max(self.px, soft), 0, 1) if soft else self.cov(sdf)
        self.height = self.height * (1 - c) + h * c

    def add_height(self, sdf, dh):
        self.height += self.cov(sdf) * dh

    # ---- distance fields
    def seg_param(self, a, b):
        ax, az = a
        bx, bz = b
        pax, paz = self.X - ax, self.Z - az
        bax, baz = bx - ax, bz - az
        L2 = bax * bax + baz * baz
        h = np.clip((pax * bax + paz * baz) / L2, 0, 1)
        return np.hypot(pax - bax * h, paz - baz * h), h * np.sqrt(L2)

    def polyline(self, pts, r, dashed=False, closed=False):
        if closed:
            pts = list(pts) + [pts[0]]
        best = np.full(self.X.shape, 1e9, np.float32)
        along = np.zeros(self.X.shape, np.float32)
        acc = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            d, s = self.seg_param(a, b)
            m = d < best
            best = np.where(m, d, best)
            along = np.where(m, acc + s, along)
            acc += float(np.hypot(b[0] - a[0], b[1] - a[1]))
        sdf = best - r
        if dashed:
            on = ((along / DASH) % 1.0) < DUTY
            sdf = np.where(on, sdf, 1e9)
        return sdf

    def polygon(self, pts):
        """Signed distance to a closed polygon (negative inside)."""
        pts = np.asarray(pts, float)
        d = np.full(self.X.shape, 1e9)
        sign = np.ones(self.X.shape)
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            e = b - a
            wx, wz = self.X - a[0], self.Z - a[1]
            h = np.clip((wx * e[0] + wz * e[1]) / (e @ e), 0, 1)
            d = np.minimum(d, np.hypot(wx - e[0] * h, wz - e[1] * h))
            c1 = self.Z >= a[1]
            c2 = self.Z < b[1]
            c3 = e[0] * wz > e[1] * wx
            flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            sign = np.where(flip, -sign, sign)
        return sign * d

    def rect(self, x0, x1, z0, z1):
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        hx, hz = abs(x1 - x0) / 2, abs(z1 - z0) / 2
        qx, qz = np.abs(self.X - cx) - hx, np.abs(self.Z - cz) - hz
        return np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0)

    def stitch(self, pts, closed=False):
        sdf = self.polyline(pts, STITCH_W, dashed=True, closed=closed)
        self.paint(sdf, THREAD)
        self.add_height(sdf, 0.12)


def inset_polygon(pts, d):
    """Offset a convex CCW/CW polygon inward by d."""
    pts = np.asarray(pts, float)
    n = len(pts)
    area = 0.5 * sum(pts[i, 0] * pts[(i + 1) % n, 1] - pts[(i + 1) % n, 0] * pts[i, 1] for i in range(n))
    s = 1 if area > 0 else -1
    lines = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        e = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-e[1], e[0]]) * s           # inward normal
        lines.append((a + nrm * d, e))
    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        t = np.linalg.solve(A, p2 - p1)
        out.append(tuple(p1 + d1 * t[0]))
    return out


def bezier(p0, p1, p2, n=28):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.array(p) for p in (p0, p1, p2))
    return [tuple(v) for v in (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2]


def mirror(pts):
    return [(-x, z) for x, z in pts]


def yoke_points(side):
    pts = [(0.0, 0.912), (0.10, 0.935), (0.19, 0.955), (0.26, 0.965)]
    return pts if side > 0 else mirror(pts)


def pocket_points(side):
    # top-inner, top-outer, bottom-outer, point, bottom-inner (wearer's left)
    pts = [(0.036, 0.905), (0.160, 0.918), (0.157, 0.810), (0.099, 0.790), (0.041, 0.803)]
    return pts if side > 0 else mirror(pts)


def draw(cv):
    # ---- yoke + center back seam (felled seams: groove + double stitch)
    for side in (1, -1):
        yk = yoke_points(side)
        cv.add_height(cv.polyline(yk, 0.0007), -0.18)
        for off in (0.0025, 0.0085):
            cv.stitch([(x, z - off) for x, z in yk])
    cv.add_height(cv.polyline([(0, 0.912), (0, 0.70)], 0.0007), -0.18)
    for off in (0.002, 0.008):
        cv.stitch([(-off, 0.912 - off), (-off, 0.70)])

    # ---- back pockets
    for side in (1, -1):
        pk = pocket_points(side)
        poly = cv.polygon(pk)
        cv.lift(poly, 0.68, soft=0.0012)
        cv.paint(np.abs(poly) - 0.0011, EDGE_WEAR, 0.45)
        # top hem: double row across the opening
        a, b = np.array(pk[0]), np.array(pk[1])
        top = [tuple(a + (b - a) * 0.02), tuple(a + (b - a) * 0.98)]
        for off in (0.010, 0.016):
            cv.stitch([(x, z - off) for x, z in top])
        # sides and bottom: double row
        for d in (0.0025, 0.0085):
            ins = inset_polygon(pk, d)
            cv.stitch([ins[1], ins[2], ins[3], ins[4], ins[0]])
        # arcuate (two wings meeting at a diamond point), double row
        s = side
        for dz in (0.0, -0.0045):
            mid = (s * 0.099, 0.852 + dz)
            wing1 = bezier((s * 0.047, 0.882 + dz), (s * 0.066, 0.908 + dz), mid)
            wing2 = bezier((s * 0.151, 0.893 + dz), (s * 0.130, 0.917 + dz), mid)
            cv.stitch(wing1)
            cv.stitch(wing2)

    # ---- red tab on the wearer's right pocket, sewn into its inner edge
    pk = pocket_points(-1)
    tab = cv.rect(-0.040, -0.028, 0.862, 0.880)
    cv.lift(tab, 0.72, soft=0.0008)
    cv.paint(tab, RED_TAB)
    cv.paint(cv.polyline([(-0.0375, 0.867), (-0.031, 0.867)], 0.0005), (0.95, 0.95, 0.92), 0.8)

    # ---- belt loops: center back + one above each pocket's outer corner
    for x in (0.0, 0.168, -0.168):
        lp = cv.rect(x - 0.0062, x + 0.0062, 0.992, 1.034)
        cv.lift(lp, 0.82, soft=0.0012)
        cv.paint(np.abs(lp) - 0.0009, EDGE_WEAR, 0.5)
        cv.paint(cv.rect(x - 0.0095, x + 0.0095, 0.985, 0.992) - 0.0, (0, 0, 0), 0.25)  # shadow
        for zz in (0.995, 1.029):
            cv.stitch([(x - 0.0045, zz), (x + 0.0045, zz)])

    # ---- leather patch on the wearer's right, over the waistband
    pt = cv.rect(-0.150, -0.062, 0.987, 1.029)
    cv.lift(pt, 0.85, soft=0.0015)
    cv.paint(pt, LEATHER)
    rng = np.random.default_rng(501)
    for row, zz in enumerate(np.linspace(0.993, 1.023, 6)):        # printed text impression
        x = -0.143
        while x < -0.070:
            w = rng.uniform(0.002, 0.009)
            if row in (0, 3) or rng.random() > 0.15:
                ink = cv.polyline([(x, zz), (min(x + w, -0.069), zz)], 0.0007 if row else 0.0011)
                cv.paint(ink, LEATHER_INK, 0.85)
                cv.add_height(ink, -0.08)
            x += w + rng.uniform(0.0015, 0.004)
    cv.stitch([(-0.148, 0.989), (-0.064, 0.989), (-0.064, 1.027), (-0.148, 1.027)], closed=True)


def save(out_dir, res=RES):
    import bpy
    cv = Canvas(*res)
    draw(cv)
    paths = []
    for name, data in (("trims_color.png", cv.rgba),
                       ("trims_height.png", np.dstack([np.clip(cv.height, 0, 1)] * 3
                                                      + [np.ones_like(cv.height)]))):
        img = bpy.data.images.new(name, res[0], res[1], alpha=True)
        img.pixels.foreach_set(data.astype(np.float32).ravel())
        img.filepath_raw = os.path.join(out_dir, name)
        img.file_format = "PNG"
        img.save()
        bpy.data.images.remove(img)
        paths.append(os.path.join(out_dir, name))
    return paths


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print("saved", save(os.path.join(here, "..", "textures")))
