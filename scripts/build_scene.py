"""
Builds the "Sisyphus jeans" scene and saves pants_walk.blend.

  Blender -b --python scripts/build_scene.py            # build + save
  Blender -b --python scripts/build_scene.py -- --regen # also redraw the textures

A headless mannequin in Levi's 501-style jeans walks toward -Y, driven by a
motion-capture walk (Bandai Namco Research Motion dataset, CC BY-NC 4.0).
The Sisyphus embroidery sits on the back of the legs at the knees.
Wearer's left = +X = viewer's left from behind.
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import embroidery_art  # noqa: E402
import levis_trims  # noqa: E402

TEX = os.path.join(ROOT, "textures")
ASSETS = os.path.join(ROOT, "assets")

# ------------------------------------------------------------------ parameters
FPS = 24
MOCAP = os.path.join(ASSETS, "mocap", "walk_normal_001.bvh")
MOCAP_START = 22          # first mocap frame used (skips the start-up step)
DURATION = 120            # 5 s

# skeleton (leg lengths match the mocap so the feet stay planted)
LEG_X = 0.085
THIGH_LEN, SHIN_LEN, ANKLE_Z = 0.391, 0.410, 0.117
KNEE_Z = ANKLE_Z + SHIN_LEN
HIP_Z = KNEE_Z + THIGH_LEN
SHOULDER = (0.175, 0.0, 1.43)
UPPER_ARM, FOREARM, HAND = 0.28, 0.25, 0.17

DENIM_TINT = (0.55, 0.62, 0.95)     # multiplies the denim photo texture (1,1,1 = as scanned)
SEAT_FADE = 0.35                     # lighter worn areas on seat/thighs, 0..1
THREAD = (0.86, 0.70, 0.40)
TEE_COLOR = (0.30, 0.31, 0.33)       # heather grey t-shirt
MANNEQUIN_COLOR = (0.82, 0.80, 0.77)

# embroidery panel at the knees (2:1 artwork spanning both legs)
DESIGN_WIDTH = 0.50
DESIGN_ASPECT = 0.5
DESIGN_CENTER_Z = KNEE_Z + 0.06

# camera: orbit from left-back to right-back while following the walker
CAM_DIST, CAM_HEIGHT, CAM_TARGET_Z = 2.7, 0.85, 0.72
CAM_AZIMUTH = (42.0, -42.0)          # degrees from straight behind, + = wearer's left
CAM_LENS, CAM_FSTOP = 65, 2.8

# pants cross-section keys: z, pelvis half-width, back depth, front depth,
# blend (0 = pelvis "D" half, 1 = round leg), leg rx, leg ry back, leg ry front
PANTS_PROFILE = [
    (1.03, 0.165, 0.110, 0.100, 0.0, 0.08, 0.08, 0.08),
    (0.96, 0.180, 0.130, 0.110, 0.0, 0.08, 0.08, 0.08),
    (0.88, 0.186, 0.138, 0.110, 0.0, 0.09, 0.09, 0.09),
    (0.81, 0.180, 0.120, 0.105, 0.55, 0.088, 0.10, 0.09),
    (0.76, 0.170, 0.110, 0.100, 1.0, 0.086, 0.094, 0.086),
    (0.64, 0.170, 0.110, 0.100, 1.0, 0.077, 0.082, 0.077),
    (0.53, 0.170, 0.110, 0.100, 1.0, 0.067, 0.070, 0.066),
    (0.30, 0.170, 0.110, 0.100, 1.0, 0.063, 0.065, 0.062),
    (0.06, 0.170, 0.110, 0.100, 1.0, 0.064, 0.066, 0.063),
]
RING_N = 48
HEM_Z = PANTS_PROFILE[-1][0]

# ------------------------------------------------------------------ helpers


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def fcurves_of(id_data):
    from bpy_extras.anim_utils import action_get_channelbag_for_slot
    ad = id_data.animation_data
    bag = action_get_channelbag_for_slot(ad.action, ad.action_slot)
    return bag.fcurves if bag else []


def setattr_safe(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def loft(name, rings, cap_top=False, cap_bottom=False):
    """rings: list of lists of (x, y, z). Returns mesh object with smooth faces."""
    bm = bmesh.new()
    vr = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(rings[0])
    for r0, r1 in zip(vr[:-1], vr[1:]):
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((r0[i], r0[j], r1[j], r1[i]))
    if cap_top:
        bm.faces.new(vr[-1])
    if cap_bottom:
        bm.faces.new(vr[0])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return link(bpy.data.objects.new(name, me))


def ellipse_ring(c, rx, ry, z, n=32):
    return [(c[0] + rx * math.cos(2 * math.pi * i / n), c[1] + ry * math.sin(2 * math.pi * i / n), z)
            for i in range(n)]


def limb(name, length, radii, n=24):
    """Capsule-ish tube along -Z from 0 to -length; radii = [(t, r), ...]."""
    rings = [ellipse_ring((0, 0), max(r, 0.002), max(r, 0.002) * 0.92, -t * length, n) for t, r in radii]
    return loft(name, rings, cap_top=True, cap_bottom=True)


# ------------------------------------------------------------------ pants mesh


def ray_ellipse(cx, c, s, a, b):
    A = c * c / (a * a) + s * s / (b * b)
    B = 2 * cx * c / (a * a)
    C = cx * cx / (a * a) - 1
    return (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)


def ring_radius(theta, side, key):
    z, pa, pb_back, pb_front, blend, rx, ry_back, ry_front = key
    cx = side * LEG_X
    c, s = math.cos(theta), math.sin(theta)
    r_d = ray_ellipse(cx, c, s, pa, pb_back if s > 0 else pb_front)
    if side * c < -1e-6:
        r_d = min(r_d, -cx / c)
    ry = ry_back if s > 0 else ry_front
    r_e = 1.0 / math.sqrt(c * c / (rx * rx) + s * s / (ry * ry))
    return lerp(r_d, r_e, blend)


def interp_profile(z):
    keys = PANTS_PROFILE
    for k0, k1 in zip(keys[:-1], keys[1:]):
        if k1[0] <= z <= k0[0]:
            t = (k0[0] - z) / (k0[0] - k1[0])
            return tuple(lerp(a, b, t) for a, b in zip(k0, k1))
    return keys[0] if z > keys[0][0] else keys[-1]


def build_pants():
    top = PANTS_PROFILE[0][0]
    zs = []
    z = top
    while z > HEM_Z:
        zs.append(z)
        z -= 0.015 if z > 0.45 else 0.03
    zs.append(HEM_Z)

    bm = bmesh.new()
    vinfo = []        # per vertex: (u fraction around leg, ring circumference, is_side_seam, side)
    for side in (1, -1):
        rings = []
        for z in zs:
            key = interp_profile(z)
            pts = []
            for i in range(RING_N):
                th = 2 * math.pi * i / RING_N
                r = ring_radius(th, side, key)
                pts.append((side * LEG_X + r * math.cos(th), r * math.sin(th), z))
            circ = sum(math.dist(pts[i], pts[(i + 1) % RING_N]) for i in range(RING_N))
            ring = []
            seam_i = 0 if side > 0 else RING_N // 2          # outermost point of each leg
            for i, p in enumerate(pts):
                ring.append(bm.verts.new(p))
                th = 2 * math.pi * i / RING_N
                vinfo.append((((th + math.pi / 2) / (2 * math.pi)) % 1.0, circ, i == seam_i, side))
            rings.append(ring)
        for r0, r1 in zip(rings[:-1], rings[1:]):
            for i in range(RING_N):
                j = (i + 1) % RING_N
                bm.faces.new((r0[i], r0[j], r1[j], r1[i]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("Pants")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True

    # DesignUV: planar back projection (x, z) in meters, for trims and embroidery
    # FabricUV: unwrapped around each leg (arc length, z) in meters, for the denim weave
    # EmbUV: like DesignUV but x measured as arc length around the leg from its back
    #        center, so the embroidery wraps the leg like stitched fabric instead of smearing
    design = me.uv_layers.new(name="DesignUV")
    fabric = me.uv_layers.new(name="FabricUV")
    emb = me.uv_layers.new(name="EmbUV")
    for poly in me.polygons:
        fr = [vinfo[me.loops[li].vertex_index][0] for li in poly.loop_indices]
        wrap = max(fr) - min(fr) > 0.5
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            co = me.vertices[vi].co
            f, circ, _, side = vinfo[vi]
            if wrap and f < 0.5:
                f += 1.0
            design.data[li].uv = (co.x, co.z)
            fabric.data[li].uv = (f * circ, co.z)
            emb.data[li].uv = (side * LEG_X + (0.5 - f) * circ, co.z)
    me.uv_layers.active = fabric

    back = me.attributes.new("back_mask", "FLOAT", "POINT")
    seam = me.attributes.new("side_seam", "FLOAT", "POINT")
    leg = me.attributes.new("leg_side", "FLOAT", "POINT")
    for v in me.vertices:
        leg.data[v.index].value = vinfo[v.index][3]
        back.data[v.index].value = smoothstep(-0.25, 0.25, v.normal.y)
        seam.data[v.index].value = 1.0 if vinfo[v.index][2] and v.co.z < 0.99 else 0.0

    obj = link(bpy.data.objects.new("Pants", me))
    groups = {n: obj.vertex_groups.new(name=n)
              for n in ("pelvis", "thigh.L", "shin.L", "thigh.R", "shin.R")}
    half = len(me.vertices) // 2
    for v in me.vertices:
        x, z = v.co.x, v.co.z
        sfx = ".L" if v.index < half else ".R"
        w_pel = smoothstep(0.80, 0.99, z)
        seam_w = max(0.0, 1 - abs(x) / 0.06) * smoothstep(0.74, 0.84, z)
        w_pel = max(w_pel, 0.85 * seam_w)
        w_shin = smoothstep(KNEE_Z + 0.05, KNEE_Z - 0.06, z)
        w_thigh = (1 - w_shin) * (1 - w_pel)
        w_shin *= (1 - w_pel)
        for name, w in (("pelvis", w_pel), ("thigh" + sfx, w_thigh), ("shin" + sfx, w_shin)):
            if w > 1e-4:
                groups[name].add([v.index], w, "REPLACE")
    return obj


# ------------------------------------------------------------------ body


def build_torso():
    """Headless mannequin torso; the tee's hem sits just above the waistband."""
    keys = [(0.97, 0.150, 0.098), (1.036, 0.154, 0.100),
            (1.040, 0.172, 0.121), (1.10, 0.168, 0.115), (1.20, 0.165, 0.110),
            (1.30, 0.173, 0.112), (1.38, 0.186, 0.112), (1.42, 0.192, 0.108),
            (1.45, 0.186, 0.102), (1.475, 0.165, 0.094), (1.495, 0.135, 0.084),
            (1.508, 0.095, 0.070), (1.515, 0.060, 0.056),
            (1.53, 0.054, 0.052), (1.62, 0.050, 0.050)]
    obj = loft("Torso", [ellipse_ring((0, 0), rx, ry, z) for z, rx, ry in keys], cap_top=True)
    for p in obj.data.polygons:
        tee = 1.038 < p.center.z < 1.512
        p.material_index = 0 if tee else 1
    obj.modifiers.new("Subsurf", "SUBSURF").levels = 2
    return obj


def build_arm_parts(sfx):
    up = limb("UpperArm" + sfx, UPPER_ARM, [(-0.12, 0.0), (-0.1, 0.03), (0.0, 0.046), (0.5, 0.042),
                                              (1.0, 0.036), (1.1, 0.0)])
    sleeve = loft("Sleeve" + sfx, [ellipse_ring((0, 0), r, r * 0.9, -t, 32)
                                   for t, r in ((-0.03, 0.060), (0.05, 0.058), (0.13, 0.054))])
    sleeve.modifiers.new("Solidify", "SOLIDIFY").thickness = 0.004
    fore = limb("Forearm" + sfx, FOREARM, [(-0.1, 0.0), (-0.08, 0.026), (0.0, 0.037), (0.35, 0.036),
                                            (1.0, 0.025), (1.08, 0.0)])
    hand = limb("Hand" + sfx, HAND, [(-0.08, 0.0), (-0.06, 0.02), (0.05, 0.028), (0.5, 0.034),
                                      (0.85, 0.026), (1.0, 0.0)])
    hand.scale = (1.0, 0.45, 1.0)          # flat mitten hand
    for o in (up, sleeve, fore, hand):
        o.modifiers.new("Subsurf", "SUBSURF").levels = 1
    return [("upperarm" + sfx, up), ("upperarm" + sfx, sleeve), ("forearm" + sfx, fore),
            ("hand" + sfx, hand)]


# sneaker sections along the foot (y relative to the ankle): y, width, upper height
SHOE_SECTIONS = [(0.062, 0.030, 0.050), (0.055, 0.062, 0.072), (0.035, 0.078, 0.080),
                 (0.0, 0.084, 0.088), (-0.05, 0.092, 0.085), (-0.10, 0.100, 0.066),
                 (-0.15, 0.098, 0.050), (-0.19, 0.084, 0.040), (-0.212, 0.056, 0.030),
                 (-0.222, 0.020, 0.020)]
SOLE_H = 0.026


def _shoe_ring(w, h, base, n=24, flat=False):
    """Closed cross-section: dome of width w and height h on a flat bottom at z=base."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        c, s = math.cos(a), math.sin(a)
        if flat:     # rounded rectangle (sole)
            x = w / 2 * math.copysign(abs(c) ** 0.35, c)
            z = base + h / 2 + h / 2 * math.copysign(abs(s) ** 0.35, s)
        else:        # dome: upper half is an ellipse arc, lower half squashed flat
            x = w / 2 * c
            z = base + (h * s if s > 0 else 0.15 * h * s)
        pts.append((x, z))
    return pts


def build_shoe(sfx, arm):
    """Lofted sneaker (rubber sole + leather upper) skinned to foot + toe so it flexes."""
    x0 = LEG_X if sfx == ".L" else -LEG_X
    objs = []
    for name in ("Sole", "Upper"):
        rings = []
        for y, w, h in SHOE_SECTIONS:
            if name == "Sole":
                prof = _shoe_ring(w + 0.012, SOLE_H, 0.0, flat=True)
            else:
                prof = _shoe_ring(w, h, SOLE_H - 0.004)
            rings.append([(x0 + px, y, pz) for px, pz in prof])
        o = loft("Shoe" + name + sfx, rings, cap_top=True, cap_bottom=True)
        me = o.data
        foot, toe = o.vertex_groups.new(name="foot" + sfx), o.vertex_groups.new(name="toe" + sfx)
        for v in me.vertices:
            w = smoothstep(-0.10, -0.15, v.co.y)
            foot.add([v.index], 1 - w, "REPLACE")
            toe.add([v.index], w, "REPLACE")
        o.parent = arm
        o.modifiers.new("Armature", "ARMATURE").object = arm
        o.modifiers.new("Subsurf", "SUBSURF").levels = 2
        objs.append(o)
    return objs


# ------------------------------------------------------------------ rig


def build_rig():
    arm_data = bpy.data.armatures.new("Rig")
    arm = link(bpy.data.objects.new("Rig", arm_data))
    arm.show_in_front = True
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    def bone(name, head, tail, parent=None, connect=False):
        b = eb.new(name)
        b.head, b.tail = head, tail
        b.roll = 0
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        return b

    bone("pelvis", (0, 0, HIP_Z), (0, 0, HIP_Z + 0.2))
    bone("spine", (0, 0, HIP_Z + 0.2), (0, 0, 1.50), "pelvis", True)
    for sfx, s in ((".L", 1), (".R", -1)):
        x = s * LEG_X
        bone("thigh" + sfx, (x, 0, HIP_Z), (x, 0, KNEE_Z), "pelvis")
        bone("shin" + sfx, (x, 0, KNEE_Z), (x, 0, ANKLE_Z), "thigh" + sfx, True)
        bone("foot" + sfx, (x, 0, ANKLE_Z), (x, -0.13, 0.03), "shin" + sfx, True)
        bone("toe" + sfx, (x, -0.13, 0.03), (x, -0.22, 0.02), "foot" + sfx, True)
        sx, sy, sz = SHOULDER
        elbow = (s * (sx + 0.01), sy, sz - UPPER_ARM)
        wrist = (s * (sx + 0.015), sy, sz - UPPER_ARM - FOREARM)
        bone("upperarm" + sfx, (s * sx, sy, sz), elbow, "spine")
        bone("forearm" + sfx, elbow, wrist, "upperarm" + sfx, True)
        bone("hand" + sfx, wrist, (wrist[0], wrist[1], wrist[2] - HAND), "forearm" + sfx, True)
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return arm


def parent_to_bone(obj, arm, bone_name, world=None):
    b = arm.data.bones[bone_name]
    mw = world if world is not None else Matrix.LocRotScale(obj.location, obj.rotation_euler, obj.scale)
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    tail_mat = arm.matrix_world @ b.matrix_local @ Matrix.Translation((0, b.length, 0))
    obj.matrix_parent_inverse = tail_mat.inverted()
    obj.matrix_basis = mw


def limb_world(bone, scale):
    """World matrix for a limb modeled along -Z from the origin, placed on a bone's head."""
    head = bone.head_local
    d = (bone.tail_local - head).normalized()
    rot = Vector((0, 0, -1)).rotation_difference(d).to_matrix().to_4x4()
    return Matrix.Translation(head) @ rot @ Matrix.Diagonal((*scale, 1))


# ------------------------------------------------------------------ mocap retarget

# our bone -> mocap joints (from, to) whose direction it copies
DIR_MAP = {
    "thigh.L": ("UpperLeg_L", "LowerLeg_L"), "shin.L": ("LowerLeg_L", "Foot_L"), "foot.L": ("Foot_L", "Toes_L"),
    "thigh.R": ("UpperLeg_R", "LowerLeg_R"), "shin.R": ("LowerLeg_R", "Foot_R"), "foot.R": ("Foot_R", "Toes_R"),
    "upperarm.L": ("UpperArm_L", "LowerArm_L"), "forearm.L": ("LowerArm_L", "Hand_L"), "hand.L": ("Hand_L", "Hand_L:tail"),
    "upperarm.R": ("UpperArm_R", "LowerArm_R"), "forearm.R": ("LowerArm_R", "Hand_R"), "hand.R": ("Hand_R", "Hand_R:tail"),
}
TOE_MIN_PITCH = math.radians(-6)   # toes stay ~flat while the heel lifts
ORDER = ["pelvis", "spine", "thigh.L", "shin.L", "foot.L", "toe.L", "thigh.R", "shin.R", "foot.R", "toe.R",
         "upperarm.L", "forearm.L", "hand.L", "upperarm.R", "forearm.R", "hand.R"]


def import_mocap():
    scn = bpy.context.scene
    bpy.ops.import_anim.bvh(filepath=MOCAP, global_scale=0.01, axis_forward="-Z", axis_up="Y",
                            use_fps_scale=True, update_scene_fps=False, update_scene_duration=False)
    mc = bpy.context.object
    mc.name = "Mocap_Source"
    mc.hide_render = True
    return mc


def frame3(lat, up):
    """Rotation whose X = lateral (wearer's left), Z = up, Y = backward."""
    lat = lat.normalized()
    up = (up - lat * up.dot(lat)).normalized()
    back = up.cross(lat)
    return Matrix((lat, back, up)).transposed()


def retarget(arm, mc):
    scn = bpy.context.scene
    W = mc.matrix_world
    mp = mc.pose.bones

    def J(name):
        if name.endswith(":tail"):
            return W @ mp[name[:-5]].tail
        return W @ mp[name].head

    rest = {b.name: b.matrix_local.copy() for b in arm.data.bones}
    parent = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    prev_q = {}
    hips = []
    for i in range(DURATION):
        scn.frame_set(MOCAP_START + i)
        hc = (J("UpperLeg_L") + J("UpperLeg_R")) / 2
        hips.append(hc.copy())
        des = {}
        for name in ORDER:
            r = rest[name]
            r_rot = r.to_3x3()
            p = parent[name]
            if name == "pelvis":
                R = frame3(J("UpperLeg_L") - J("UpperLeg_R"), J("Spine") - hc)
                head = hc
            elif name == "spine":
                R = frame3(J("UpperArm_L") - J("UpperArm_R"), J("Neck") - J("Spine"))
                head = des[p] @ rest[p].inverted() @ r.translation
            else:
                pr = des[p].to_3x3() @ rest[p].to_3x3().inverted()      # parent's world delta
                head = des[p] @ rest[p].inverted() @ r.translation
                if name.startswith("toe"):
                    fd = (des[p].to_3x3() @ Vector((0, 1, 0))).normalized()   # foot direction
                    pitch = max(math.asin(max(-1, min(1, fd.z))), TOE_MIN_PITCH)
                    fwd = des["pelvis"].to_3x3() @ Vector((0, -1, 0))
                    hz = Vector((fwd.x, fwd.y, 0)).normalized()
                    d = hz * math.cos(pitch) + Vector((0, 0, math.sin(pitch)))
                else:
                    a, b = DIR_MAP[name]
                    d = (J(b) - J(a)).normalized()
                cur = pr @ (r_rot @ Vector((0, 1, 0)))
                R = cur.rotation_difference(d).to_matrix() @ pr
            des[name] = Matrix.Translation(head) @ (R @ r_rot).to_4x4()

        frame = 1 + i
        for name in ORDER:
            p = parent[name]
            if p is None:
                basis = rest[name].inverted() @ des[name]
            else:
                rel = rest[p].inverted() @ rest[name]
                basis = rel.inverted() @ des[p].inverted() @ des[name]
            loc, q, _ = basis.decompose()
            if name in prev_q and q.dot(prev_q[name]) < 0:
                q.negate()
            prev_q[name] = q
            pb = arm.pose.bones[name]
            pb.rotation_quaternion = q
            pb.keyframe_insert("rotation_quaternion", frame=frame)
            if p is None:
                pb.location = loc
                pb.keyframe_insert("location", frame=frame)
    return hips


# ------------------------------------------------------------------ materials


def img_node(nt, path, color=True, loc=(0, 0), extension="REPEAT"):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(path, check_existing=True)
    if not color:
        n.image.colorspace_settings.name = "Non-Color"
    n.extension = extension
    n.location = loc
    return n


def mix(nt, fac, a, b, loc=(0, 0), data_type="RGBA", blend="MIX"):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = data_type
    if data_type == "RGBA":
        n.blend_type = blend
    n.location = loc
    ins = [s for s in n.inputs if s.enabled]
    for sock, val in zip(ins, (fac, a, b)):
        if isinstance(val, bpy.types.NodeSocket):
            nt.links.new(val, sock)
        else:
            sock.default_value = val
    return [s for s in n.outputs if s.enabled][0]


def math_node(nt, op, a, b=None, loc=(0, 0), clamp=False, c=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    n.location = loc
    for sock, val in zip(n.inputs, (a, b, c)):
        if val is None:
            continue
        if isinstance(val, bpy.types.NodeSocket):
            nt.links.new(val, sock)
        else:
            sock.default_value = val
    return n.outputs[0]


def placement(nt, uv_out, x_scale, z_scale, x_center, z0, label, loc):
    """Maps planar (x, z) meters to image UV: u = 0.5 - (x - x_center)/x_scale, v = (z - z0)/z_scale."""
    m = nt.nodes.new("ShaderNodeMapping")
    m.name = m.label = label
    m.location = loc
    m.inputs["Scale"].default_value = (-1 / x_scale, 1 / z_scale, 1)
    m.inputs["Location"].default_value = (0.5 + x_center / x_scale, -z0 / z_scale, 0)
    nt.links.new(uv_out, m.inputs["Vector"])
    return m.outputs["Vector"]


def denim_material(paths):
    mat = bpy.data.materials.new("Denim_501")
    nt = mat.node_tree
    nodes, L = nt.nodes, nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (2200, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (1900, 0)
    bsdf.inputs["Sheen Weight"].default_value = 0.25
    bsdf.inputs["Sheen Roughness"].default_value = 0.6

    fab_uv = nodes.new("ShaderNodeUVMap")
    fab_uv.uv_map = "FabricUV"
    fab_uv.location = (-1800, -400)
    des_uv = nodes.new("ShaderNodeUVMap")
    des_uv.uv_map = "DesignUV"
    des_uv.location = (-1800, 300)
    geo_z = nodes.new("ShaderNodeSeparateXYZ")
    geo_z.location = (-1550, 600)
    L.new(des_uv.outputs["UV"], geo_z.inputs[0])
    z = geo_z.outputs["Y"]      # DesignUV.y = rest height in meters

    # --- denim photo texture (Poly Haven denim_fabric_06, 25.4 cm tile)
    fmap = nodes.new("ShaderNodeMapping")
    fmap.location = (-1550, -400)
    fmap.inputs["Scale"].default_value = (1 / 0.254, 1 / 0.254, 1)
    L.new(fab_uv.outputs["UV"], fmap.inputs["Vector"])
    diff = img_node(nt, paths["denim_diff"], True, (-1300, -250))
    nrm = img_node(nt, paths["denim_nor"], False, (-1300, -550))
    rough = img_node(nt, paths["denim_rough"], False, (-1300, -850))
    for n in (diff, nrm, rough):
        L.new(fmap.outputs["Vector"], n.inputs["Vector"])
    denim = mix(nt, 1.0, diff.outputs["Color"], (*DENIM_TINT, 1), (-1000, -250), blend="MULTIPLY")

    # worn lighter areas: seat + upper thighs (back) with blotchy noise
    attr_back = nodes.new("ShaderNodeAttribute")
    attr_back.attribute_name = "back_mask"
    attr_back.location = (-1300, 900)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 9
    noise.inputs["Detail"].default_value = 4
    noise.location = (-1300, 650)
    L.new(des_uv.outputs["UV"], noise.inputs["Vector"])
    low = math_node(nt, "MULTIPLY", math_node(nt, "SUBTRACT", z, 0.66), 3.0, clamp=True)
    high = math_node(nt, "SUBTRACT", 1.0, math_node(nt, "MULTIPLY", math_node(nt, "SUBTRACT", z, 0.93), 8.0, clamp=True))
    fade = math_node(nt, "MULTIPLY", math_node(nt, "MULTIPLY", low, high), attr_back.outputs["Fac"])
    fade = math_node(nt, "MULTIPLY", fade, math_node(nt, "MULTIPLY", noise.outputs["Fac"], 1.4))
    fade = math_node(nt, "MULTIPLY", fade, SEAT_FADE, clamp=True)
    denim = mix(nt, fade, denim, (0.30, 0.42, 0.62, 1), (-50, -100), blend="SCREEN")

    # --- waistband + hem: stitch rings all around (computed from rest height)
    def ring_line(z_at, half_w):
        d = math_node(nt, "ABSOLUTE", math_node(nt, "SUBTRACT", z, z_at))
        return math_node(nt, "LESS_THAN", d, half_w)

    fab_sep = nodes.new("ShaderNodeSeparateXYZ")
    fab_sep.location = (-1300, -1100)
    L.new(fab_uv.outputs["UV"], fab_sep.inputs[0])
    dash = math_node(nt, "LESS_THAN", math_node(nt, "FRACT", math_node(nt, "MULTIPLY", fab_sep.outputs["X"], 290.0)), 0.72)
    dash_z = math_node(nt, "LESS_THAN", math_node(nt, "FRACT", math_node(nt, "MULTIPLY", z, 290.0)), 0.72)
    rings = None
    for z_at in (0.9985, 1.0255, HEM_Z + 0.016):
        r = ring_line(z_at, 0.0006)
        rings = r if rings is None else math_node(nt, "MAXIMUM", rings, r)
    ring_stitch = math_node(nt, "MULTIPLY", rings, dash)
    # side seams (outseam) on each leg
    attr_seam = nodes.new("ShaderNodeAttribute")
    attr_seam.attribute_name = "side_seam"
    attr_seam.location = (-1300, -1300)
    seam_line = math_node(nt, "GREATER_THAN", attr_seam.outputs["Fac"], 0.86)
    seam_stitch = math_node(nt, "MULTIPLY", seam_line, dash_z)
    stitch = math_node(nt, "MAXIMUM", ring_stitch, seam_stitch, clamp=True)
    color = mix(nt, stitch, denim, (*THREAD, 1), (200, -100))
    band_edge = ring_line(0.9935, 0.0012)
    hem_edge = ring_line(HEM_Z + 0.03, 0.0015)

    # --- Levi's trims (back): pockets, yoke, loops, patch, red tab
    tuv = placement(nt, des_uv.outputs["UV"], 2 * levis_trims.X_HALF, levis_trims.Z1 - levis_trims.Z0,
                    0.0, levis_trims.Z0, "Trims Placement", (-1300, 1300))
    tcol = img_node(nt, paths["trims_color"], True, (-1000, 1400), "CLIP")
    thgt = img_node(nt, paths["trims_height"], False, (-1000, 1100), "EXTEND")
    for n in (tcol, thgt):
        L.new(tuv, n.inputs["Vector"])
    tmask = math_node(nt, "MULTIPLY", tcol.outputs["Alpha"], attr_back.outputs["Fac"], (-700, 1400))
    color = mix(nt, tmask, color, tcol.outputs["Color"], (400, 300))

    # --- Sisyphus embroidery at the knees
    h = DESIGN_WIDTH * DESIGN_ASPECT
    emb_uv = nodes.new("ShaderNodeUVMap")
    emb_uv.uv_map = "EmbUV"
    emb_uv.location = (-1800, 1900)
    euv = placement(nt, emb_uv.outputs["UV"], DESIGN_WIDTH, h, 0.0, DESIGN_CENTER_Z - h / 2,
                    "Embroidery Placement", (-1300, 1900))
    ecol = img_node(nt, paths["embroidery"], True, (-1000, 1900), "CLIP")
    ecol.label = "Embroidery Art"
    L.new(euv, ecol.inputs["Vector"])
    # keep each half of the artwork on its own leg
    attr_leg = nodes.new("ShaderNodeAttribute")
    attr_leg.attribute_name = "leg_side"
    attr_leg.location = (-1300, 2150)
    emb_sep = nodes.new("ShaderNodeSeparateXYZ")
    L.new(emb_uv.outputs["UV"], emb_sep.inputs[0])
    own_leg = math_node(nt, "GREATER_THAN", math_node(nt, "MULTIPLY", emb_sep.outputs["X"], attr_leg.outputs["Fac"]), 0.0)
    emask = math_node(nt, "MULTIPLY", ecol.outputs["Alpha"], own_leg, (-700, 1900))
    color = mix(nt, emask, color, ecol.outputs["Color"], (600, 300))
    L.new(color, bsdf.inputs["Base Color"])

    satin = nodes.new("ShaderNodeTexWave")
    satin.wave_type = "BANDS"
    satin.bands_direction = "DIAGONAL"
    satin.inputs["Scale"].default_value = 260
    satin.location = (-700, 1700)
    L.new(euv, satin.inputs["Vector"])

    thread_mask = math_node(nt, "MAXIMUM", stitch, emask)
    rgh = mix(nt, thread_mask, rough.outputs["Color"], (0.55, 0.55, 0.55, 1), (400, -500))
    L.new(rgh, bsdf.inputs["Roughness"])

    # --- normals: denim normal map, then bump for trims/embroidery/stitches/edges
    nmap = nodes.new("ShaderNodeNormalMap")
    nmap.inputs["Strength"].default_value = 1.2
    nmap.location = (-1000, -550)
    L.new(nrm.outputs["Color"], nmap.inputs["Color"])
    trims_h = math_node(nt, "MULTIPLY", math_node(nt, "SUBTRACT", thgt.outputs["Color"], 0.5), attr_back.outputs["Fac"])
    emb_h = math_node(nt, "MULTIPLY", emask, math_node(nt, "MULTIPLY_ADD", satin.outputs["Fac"], 0.3, c=0.7))
    height = math_node(nt, "ADD", trims_h, math_node(nt, "MULTIPLY", emb_h, 0.5))
    height = math_node(nt, "ADD", height, math_node(nt, "MULTIPLY", stitch, 0.08))
    height = math_node(nt, "SUBTRACT", height, math_node(nt, "MULTIPLY", math_node(nt, "MAXIMUM", band_edge, hem_edge), 0.25))
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = 0.004
    bump.location = (1500, -500)
    L.new(height, bump.inputs["Height"])
    L.new(nmap.outputs["Normal"], bump.inputs["Normal"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def simple_material(name, color, rough=0.5, sheen=0.0, coat=0.0):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Sheen Weight"].default_value = sheen
    bsdf.inputs["Coat Weight"].default_value = coat
    return mat


def cotton_material():
    mat = simple_material("Tee_Cotton", TEE_COLOR, 0.95, sheen=0.6)
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 900
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.15
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def floor_material(paths):
    mat = bpy.data.materials.new("Concrete")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / 3.0, 1 / 3.0, 1)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    d = img_node(nt, paths["floor_diff"], True)
    n = img_node(nt, paths["floor_nor"], False)
    r = img_node(nt, paths["floor_rough"], False)
    for t in (d, n, r):
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(n.outputs["Color"], nm.inputs["Color"])
    nt.links.new(d.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(mix(nt, 0.55, r.outputs["Color"], (1, 1, 1, 1)), bsdf.inputs["Roughness"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# ------------------------------------------------------------------ scene


def textures(regen):
    paths = {
        "embroidery": os.path.join(TEX, "embroidery.png"),
        "trims_color": os.path.join(TEX, "trims_color.png"),
        "trims_height": os.path.join(TEX, "trims_height.png"),
        "denim_diff": os.path.join(ASSETS, "tex", "denim_fabric_06_diff_2k.jpg"),
        "denim_nor": os.path.join(ASSETS, "tex", "denim_fabric_06_nor_gl_2k.jpg"),
        "denim_rough": os.path.join(ASSETS, "tex", "denim_fabric_06_rough_2k.jpg"),
        "floor_diff": os.path.join(ASSETS, "tex", "concrete_floor_worn_001_diff_2k.jpg"),
        "floor_nor": os.path.join(ASSETS, "tex", "concrete_floor_worn_001_nor_gl_2k.jpg"),
        "floor_rough": os.path.join(ASSETS, "tex", "concrete_floor_worn_001_rough_2k.jpg"),
        "hdri": os.path.join(ASSETS, "hdri", "german_town_street_2k.hdr"),
    }

    def stale(png, src):
        return regen or not os.path.exists(png) or os.path.getmtime(png) < os.path.getmtime(src)

    if stale(paths["embroidery"], embroidery_art.__file__):
        embroidery_art.save_png(paths["embroidery"])
    if stale(paths["trims_color"], levis_trims.__file__):
        levis_trims.save(TEX)
    return paths


def add_camera(name, target, rig, loc):
    cam = link(bpy.data.objects.new(name, bpy.data.cameras.new(name)))
    cam.data.lens = CAM_LENS
    cam.data.dof.use_dof = True
    cam.data.dof.focus_object = target
    cam.data.dof.aperture_fstop = CAM_FSTOP
    cam.parent = rig
    cam.location = loc
    c = cam.constraints.new("TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    return cam


def orbit_pos(az_deg):
    a = math.radians(az_deg)
    return (CAM_DIST * math.sin(a), CAM_DIST * math.cos(a), CAM_HEIGHT)


def build_camera(hips):
    scn = bpy.context.scene
    # smooth follow: straight-line fit through the hip path
    n = len(hips)
    tm = (n - 1) / 2
    ym = sum(h.y for h in hips) / n
    xm = sum(h.x for h in hips) / n
    vy = sum((t - tm) * (h.y - ym) for t, h in enumerate(hips)) / sum((t - tm) ** 2 for t in range(n))

    rig = link(bpy.data.objects.new("CameraRig", None))
    target = link(bpy.data.objects.new("CAM_Target", None))
    target.parent = rig
    target.location = (0, 0, CAM_TARGET_Z)
    rig.location = (xm, ym - vy * tm, 0)
    rig.keyframe_insert("location", frame=1)
    rig.location = (xm, ym + vy * tm, 0)
    rig.keyframe_insert("location", frame=n)

    cam = add_camera("CAM_Orbit", target, rig, orbit_pos(CAM_AZIMUTH[0]))
    for i in range(0, n, 4):
        cam.location = orbit_pos(lerp(CAM_AZIMUTH[0], CAM_AZIMUTH[1], smoothstep(0, n - 1, i)))
        cam.keyframe_insert("location", frame=1 + i)
    for fc in fcurves_of(rig):
        fc.extrapolation = "LINEAR"
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    scn.camera = cam

    # fixed-angle cameras on the same follow rig
    for name, az in (("CAM_LeftBack", CAM_AZIMUTH[0]), ("CAM_Back", 0.0), ("CAM_RightBack", CAM_AZIMUTH[1])):
        add_camera(name, target, rig, orbit_pos(az))


def build(regen=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.fps = FPS
    scn.render.resolution_x, scn.render.resolution_y = 1080, 1920
    scn.render.engine = "BLENDER_EEVEE"
    scn.view_settings.view_transform = "AgX"
    try:
        scn.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    ee = scn.eevee
    setattr_safe(ee, "use_raytracing", True)
    setattr_safe(ee, "use_shadows", True)
    setattr_safe(ee, "use_fast_gi", True)
    ee.taa_render_samples = 64
    scn.render.use_motion_blur = True
    scn.render.motion_blur_shutter = 0.5

    paths = textures(regen)

    arm = build_rig()
    pants = build_pants()
    pants.data.materials.append(denim_material(paths))
    pants.parent = arm
    am = pants.modifiers.new("Armature", "ARMATURE")
    am.object = arm
    sol = pants.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.005
    sol.offset = 1.0
    pants.modifiers.new("Subsurf", "SUBSURF").levels = 2

    skin = simple_material("Mannequin", MANNEQUIN_COLOR, 0.38, coat=0.2)
    tee = cotton_material()
    torso = build_torso()
    torso.data.materials.append(tee)
    torso.data.materials.append(skin)
    parent_to_bone(torso, arm, "spine")
    rubber = simple_material("Rubber", (0.88, 0.87, 0.84), 0.7)
    leather = simple_material("ShoeLeather", (0.92, 0.91, 0.89), 0.45)
    for sfx in (".L", ".R"):
        for bone_name, part in build_arm_parts(sfx):
            part.data.materials.append(tee if part.name.startswith("Sleeve") else skin)
            parent_to_bone(part, arm, bone_name, limb_world(arm.data.bones[bone_name], part.scale))
        sole, upper = build_shoe(sfx, arm)
        sole.data.materials.append(rubber)
        upper.data.materials.append(leather)

    mc = import_mocap()
    hips = retarget(arm, mc)
    mc.hide_set(True)
    scn.frame_start, scn.frame_end = 1, DURATION

    bpy.ops.mesh.primitive_plane_add(size=1)
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.scale = (80, 80, 1)
    floor.data.materials.append(floor_material(paths))

    build_camera(hips)

    world = bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    wn = world.node_tree.nodes
    env = wn.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(paths["hdri"])
    mapn = wn.new("ShaderNodeMapping")
    mapn.inputs["Rotation"].default_value = (0, 0, math.radians(140))
    tc = wn.new("ShaderNodeTexCoord")
    world.node_tree.links.new(tc.outputs["Generated"], mapn.inputs["Vector"])
    world.node_tree.links.new(mapn.outputs["Vector"], env.inputs["Vector"])
    world.node_tree.links.new(env.outputs["Color"], wn["Background"].inputs["Color"])
    wn["Background"].inputs["Strength"].default_value = 1.0

    sun = link(bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN")))
    sun.data.energy = 2.5
    sun.data.angle = math.radians(3)
    sun.rotation_euler = (math.radians(48), math.radians(-20), math.radians(150))

    if hasattr(scn.render.image_settings, "media_type"):
        scn.render.image_settings.media_type = "VIDEO"
    scn.render.image_settings.file_format = "FFMPEG"
    scn.render.ffmpeg.format = "MPEG4"
    scn.render.ffmpeg.codec = "H264"
    scn.render.ffmpeg.constant_rate_factor = "HIGH"
    scn.render.filepath = "//render/walk_"
    scn.frame_set(1)

    path = os.path.join(ROOT, "pants_walk.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, relative_remap=True)
    print("saved", path)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    build(regen="--regen" in argv)
