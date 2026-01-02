bl_info = {
    "name": "Geo-Kineti-Cam",
    "author": "Jiovanie Velazquez",
    "version": (0, 9, 13),
    "blender": (4, 2, 0),
    "location": "View3D > Header & N-Panel",
    "description": "A buttery viewport controller for smooth kinetic based navigation.",
    "category": "View",
}

import bpy
import bmesh
import math
import time
from mathutils import Vector, Quaternion, Matrix, Euler
from bpy.app.handlers import persistent

# --- CONSTANTS ---
DEADZONE = 0.0005
ADDON_NAME = __package__ if __package__ else __name__
SCAN_INTERVAL = 0.1 

# --- STATE ---
def get_default_state():
    return {
        "mode": 'MANUAL',
        "target_focus": Vector((0,0,0)),
        "target_dist": 10.0,
        "target_rot": Quaternion((1,0,0,0)),
        "last_sel_hash": 0,
        "vel_pan": Vector((0,0,0)), 
        "vel_zoom": 0.0,
        "vel_rot": Quaternion((1,0,0,0)), 
        "last_shake_offset_loc": Vector((0,0,0)),
        "last_shake_offset_rot": Quaternion((1,0,0,0)),
        "last_loc": None, 
        "last_rot": None,
        "last_dist": 0.0,
        "last_is_perspective": True,
        "shake_suppressed": False, 
        "buffer_pan": [],
        "buffer_zoom": [],
        "buffer_rot": [],
        "is_coasting": False,
        "coasting_start_time": 0.0,
        "last_scan_time": 0.0
    }

_state = get_default_state()

def wipe_physics():
    _state["buffer_pan"] = []
    _state["buffer_zoom"] = []
    _state["buffer_rot"] = []
    _state["vel_pan"] = Vector((0,0,0))
    _state["vel_zoom"] = 0.0
    _state["vel_rot"] = Quaternion((1,0,0,0))
    _state["is_coasting"] = False

# --- SYNC FUNCTIONS ---
def sync_dist(self, context):
    try: context.preferences.addons[ADDON_NAME].preferences.val_dist = self.hybrid_dist
    except: pass

def sync_speed(self, context):
    try: context.preferences.addons[ADDON_NAME].preferences.val_speed = self.hybrid_speed
    except: pass

def sync_friction(self, context):
    try: context.preferences.addons[ADDON_NAME].preferences.val_friction = self.hybrid_friction
    except: pass

def sync_drift(self, context):
    try: context.preferences.addons[ADDON_NAME].preferences.val_drift = self.hybrid_drift
    except: pass

def sync_use_drift(self, context):
    try: context.preferences.addons[ADDON_NAME].preferences.val_use_drift = self.hybrid_use_drift
    except: pass

def sync_auto_pilot(self, context):
    try: context.preferences.addons[ADDON_NAME].preferences.val_auto_pilot = self.hybrid_auto_pilot
    except: pass

def sync_break(self, context):
    try: context.preferences.addons[ADDON_NAME].preferences.val_break_on_manual = self.hybrid_break_on_manual
    except: pass

def get_region_data():
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            return area, area.spaces.active.region_3d
    return None, None

def get_target_data_bmesh(context):
    if context.mode != 'EDIT_MESH': return None, None, None, None
    obj = context.edit_object
    if not obj or obj.type != 'MESH': return None, None, None, None

    try: bm = bmesh.from_edit_mesh(obj.data)
    except: return None, None, None, None
    
    sel_verts = [v for v in bm.verts if v.select]
    if not sel_verts: return None, None, None, None

    mw = obj.matrix_world
    mat_rot = mw.to_3x3()
    
    sum_co = Vector((0,0,0))
    sum_normal = Vector((0,0,0))
    sel_hash_int = 0 
    count = len(sel_verts)
    world_coords = []
    
    for v in sel_verts:
        w_co = mw @ v.co
        world_coords.append(w_co)
        sum_co += w_co
        sum_normal += v.normal
        sel_hash_int += v.index

    center = sum_co / count
    max_dist = 0.0
    for w_co in world_coords:
        d = (w_co - center).length
        if d > max_dist: max_dist = d
            
    avg_normal = mat_rot @ sum_normal
    target_quat = None
    if avg_normal.length_squared > 0.001:
        avg_normal.normalize()
        base_quat = avg_normal.to_track_quat('Z', 'Y')
        iso_angle = 0.463647 
        offset = Euler((iso_angle, 0.785398, 0.0), 'XYZ').to_quaternion()
        target_quat = base_quat @ offset

    final_hash = count + sel_hash_int
    return center, max_dist, final_hash, target_quat

def average_vec(buffer):
    if not buffer: return Vector((0,0,0))
    total = Vector((0,0,0))
    for v in buffer: total += v
    return total / len(buffer)

def average_float(buffer):
    if not buffer: return 0.0
    return sum(buffer) / len(buffer)

def average_quat(buffer):
    if not buffer: return Quaternion((1,0,0,0))
    w, x, y, z = 0.0, 0.0, 0.0, 0.0
    for q in buffer:
        w += q.w; x += q.x; y += q.y; z += q.z
    avg = Quaternion((w, x, y, z))
    avg.normalize()
    return avg

def stabilize_horizon(quat):
    mat = quat.to_matrix()
    view_z = mat.col[2] 
    
    z_tilt = abs(view_z.z)
    if z_tilt > 0.99: return quat

    view_x = mat.col[0] 
    flat_x = Vector((view_x.x, view_x.y, 0))
    
    if flat_x.length_squared < 0.001: return quat
    
    flat_x.normalize()
    new_y = view_z.cross(flat_x).normalized()
    new_mat = Matrix((flat_x, new_y, view_z)).transposed()
    stab_quat = new_mat.to_quaternion()
    
    blend_start = 0.8
    if z_tilt > blend_start:
        factor = (z_tilt - blend_start) / (0.99 - blend_start)
        return stab_quat.slerp(quat, factor)
    
    return stab_quat

def update_loop():
    try:
        if not bpy.context or not bpy.context.scene: return None 
        scene = bpy.context.scene
        if not getattr(scene, "hybrid_active", False): return None 
        
        s_dist = scene.hybrid_dist
        s_auto_pilot = scene.hybrid_auto_pilot
        s_break = scene.hybrid_break_on_manual
        s_drift = scene.hybrid_drift
        s_use_drift = scene.hybrid_use_drift
        s_friction = scene.hybrid_friction
        s_speed = scene.hybrid_speed

        area, rv3d = get_region_data()
        if not rv3d: return None 

        curr_loc = rv3d.view_location.copy()
        curr_rot = rv3d.view_rotation.copy()
        curr_dist = rv3d.view_distance
        curr_is_persp = rv3d.is_perspective

        if _state["last_loc"] is None:
            _state["last_loc"] = curr_loc
            _state["last_rot"] = curr_rot
            _state["last_dist"] = curr_dist
            _state["last_is_perspective"] = curr_is_persp
        
        diff_loc = curr_loc - _state["last_loc"]
        diff_dist = curr_dist - _state["last_dist"]
        diff_rot = curr_rot @ _state["last_rot"].inverted()
        
        speed_loc = diff_loc.length
        speed_rot = diff_rot.angle
        speed_dist = abs(diff_dist)

        mode_switched = (curr_is_persp != _state["last_is_perspective"])
        view_z = curr_rot.to_matrix().col[2]
        is_axis_aligned = (abs(view_z.x) > 0.9999 or abs(view_z.y) > 0.9999 or abs(view_z.z) > 0.9999)

        if mode_switched or is_axis_aligned:
            wipe_physics()
            _state["last_shake_offset_loc"] = Vector((0,0,0))
            _state["last_shake_offset_rot"] = Quaternion((1,0,0,0))
            _state["last_loc"] = curr_loc
            _state["last_rot"] = curr_rot
            _state["last_dist"] = curr_dist
            _state["last_is_perspective"] = curr_is_persp
            return 0.01
        
        _state["last_is_perspective"] = curr_is_persp

        now = time.time()
        if (now - _state["last_scan_time"]) > SCAN_INTERVAL:
            if s_auto_pilot and bpy.context.mode == 'EDIT_MESH':
                center, radius, sel_hash, iso_quat = get_target_data_bmesh(bpy.context)
                _state["last_scan_time"] = now
                if sel_hash and sel_hash != _state["last_sel_hash"]:
                    _state["last_sel_hash"] = sel_hash
                    _state["mode"] = 'AUTO'
                    _state["target_focus"] = center
                    _state["target_dist"] = (radius + 0.5) * s_dist
                    if iso_quat: _state["target_rot"] = iso_quat
                    else:
                        direction = center - curr_loc
                        if direction.length_squared > 0.001:
                            base_look = direction.to_track_quat('-Z', 'Y')
                            offset = Euler((0.349, 0.349, 0), 'XYZ').to_quaternion()
                            _state["target_rot"] = base_look @ offset
                        else: _state["target_rot"] = curr_rot
                    wipe_physics()

        is_moving = speed_loc > DEADZONE or speed_dist > DEADZONE or speed_rot > DEADZONE
        
        if _state["mode"] == 'AUTO' and not s_auto_pilot:
             _state["mode"] = 'MANUAL'

        if _state["mode"] == 'AUTO':
            if s_break:
                if _state["last_loc"] is not None:
                    delta_loc_fight = (curr_loc - _state["last_loc"]).length
                    delta_rot_fight = curr_rot.rotation_difference(_state["last_rot"]).angle
                    tol_loc = 0.01 + (s_drift * 0.2)
                    tol_rot = 0.008 + (s_drift * 0.1)
                    if delta_loc_fight > tol_loc or delta_rot_fight > tol_rot: 
                        _state["mode"] = 'MANUAL'

            if _state["mode"] == 'AUTO':
                speed = s_speed / 10.0
                new_loc = curr_loc.lerp(_state["target_focus"], speed)
                new_dist = curr_dist + ((_state["target_dist"] - curr_dist) * speed)
                raw_rot = curr_rot.slerp(_state["target_rot"], speed)
                new_rot = stabilize_horizon(raw_rot)
                rv3d.view_location = new_loc
                rv3d.view_distance = new_dist
                rv3d.view_rotation = new_rot

        elif _state["mode"] == 'MANUAL':
            if is_moving:
                _state["is_coasting"] = False
                _state["shake_suppressed"] = True

                _state["buffer_pan"].append(diff_loc)
                _state["buffer_zoom"].append(diff_dist)
                _state["buffer_rot"].append(diff_rot)
                if len(_state["buffer_pan"]) > 3: 
                    del _state["buffer_pan"][0]
                    del _state["buffer_zoom"][0]
                    del _state["buffer_rot"][0]
            else:
                if not _state["is_coasting"] and _state["buffer_pan"]:
                    _state["vel_pan"] = average_vec(_state["buffer_pan"])
                    _state["vel_zoom"] = average_float(_state["buffer_zoom"])
                    _state["vel_rot"] = average_quat(_state["buffer_rot"])
                    has_nrg = _state["vel_pan"].length > 0.001 or abs(_state["vel_zoom"]) > 0.001 or _state["vel_rot"].angle > 0.0001
                    if has_nrg: 
                        _state["is_coasting"] = True
                        _state["coasting_start_time"] = time.time()
                        _state["shake_suppressed"] = False

                        if _state["vel_rot"].angle > 0.003:
                            _state["vel_pan"] = Vector((0,0,0))

                    _state["buffer_pan"] = []
                    _state["buffer_zoom"] = []
                    _state["buffer_rot"] = []
            
            if _state["is_coasting"]:
                friction = 0.98 - (s_friction * 0.08)
                if friction < 0: friction = 0
                
                _state["vel_pan"] *= friction
                _state["vel_zoom"] *= friction
                identity = Quaternion((1,0,0,0))
                _state["vel_rot"] = _state["vel_rot"].slerp(identity, 1.0 - friction)
                
                if _state["vel_pan"].length < 0.001 and abs(_state["vel_zoom"]) < 0.001 and _state["vel_rot"].angle < 0.0001:
                    _state["is_coasting"] = False
                else:
                    rv3d.view_location += _state["vel_pan"]
                    
                    pred_dist = rv3d.view_distance + _state["vel_zoom"]
                    if pred_dist < 0.01:
                        rv3d.view_distance = 0.01
                        _state["vel_zoom"] = 0.0
                    else:
                        rv3d.view_distance = pred_dist

                    new_rot = _state["vel_rot"] @ rv3d.view_rotation
                    target_rot = stabilize_horizon(new_rot)
                    
                    coast_duration = time.time() - _state["coasting_start_time"]
                    
                    stab_alpha = 0.1
                    if coast_duration < 0.5:
                        stab_alpha = (coast_duration / 0.5) * 0.1
                    
                    rv3d.view_rotation = new_rot.slerp(target_rot, stab_alpha)

        if s_use_drift and s_drift > 0.001 and not _state["shake_suppressed"]:
            t = time.time()
            strength = s_drift * 0.05 
            rot_strength = strength * 0.2 
            
            n_x = (math.sin(t * 1.2) + math.cos(t * 2.1) * 0.5) * strength
            n_y = (math.cos(t * 1.4) + math.sin(t * 2.4) * 0.5) * strength
            n_z = (math.sin(t * 0.5) * 0.5) * strength
            rot_pitch = math.sin(t * 0.8) * rot_strength
            rot_yaw = math.cos(t * 1.1) * rot_strength
            rot_roll = math.sin(t * 1.6) * (rot_strength * 0.5) 
            
            curr_shake_offset_loc = Vector((n_x, n_y, n_z))
            curr_shake_offset_rot = Euler((rot_pitch, rot_yaw, rot_roll), 'XYZ').to_quaternion()
            delta_shake_loc = curr_shake_offset_loc - _state["last_shake_offset_loc"]
            delta_world_loc = rv3d.view_rotation @ delta_shake_loc
            rv3d.view_location += delta_world_loc
            delta_shake_rot = curr_shake_offset_rot @ _state["last_shake_offset_rot"].inverted()
            rv3d.view_rotation = delta_shake_rot @ rv3d.view_rotation
            _state["last_shake_offset_loc"] = curr_shake_offset_loc
            _state["last_shake_offset_rot"] = curr_shake_offset_rot
        else:
            if _state["last_shake_offset_loc"].length_squared > 0:
                 _state["last_shake_offset_loc"] = Vector((0,0,0))
                 _state["last_shake_offset_rot"] = Quaternion((1,0,0,0))

        if area: area.tag_redraw()
        
        _state["last_loc"] = rv3d.view_location.copy()
        _state["last_rot"] = rv3d.view_rotation.copy()
        _state["last_dist"] = rv3d.view_distance

        if not is_moving and not _state["is_coasting"] and (not s_use_drift or s_drift <= 0.001) and _state["mode"] == 'MANUAL':
            return 0.1

    except Exception as e:
        return None
        
    return 0.01

# --- PREFS ---
class GKC_Preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_NAME
    # Hidden Storage
    val_auto_pilot: bpy.props.BoolProperty(default=True, description="Automatically fly to and focus on selected mesh elements in Edit Mode")
    val_break_on_manual: bpy.props.BoolProperty(default=True, description="Disengage Auto-Pilot instantly when manual navigation is detected")
    val_dist: bpy.props.FloatProperty(default=3.0, description="Distance multiplier when focusing on objects (Auto-Pilot)")
    val_speed: bpy.props.FloatProperty(default=0.03, description="Interpolation speed for the camera physics and auto-focus")
    val_friction: bpy.props.FloatProperty(default=0.12, description="Drag/Damping for manual camera movement (Lower = more slide)")
    val_use_drift: bpy.props.BoolProperty(default=True, description="Enable idle camera movement")
    val_drift: bpy.props.FloatProperty(default=0.2, description="Intensity of the idle camera motion (Subtle floating effect)")

# --- UI & HANDLERS ---
@persistent
def load_prefs_to_scene(dummy):
    try:
        prefs = bpy.context.preferences.addons.get(ADDON_NAME)
        scene = bpy.context.scene
        scene.hybrid_active = False 
        if prefs:
            p = prefs.preferences
            scene.hybrid_dist = p.val_dist
            scene.hybrid_speed = p.val_speed
            scene.hybrid_friction = p.val_friction
            scene.hybrid_use_drift = p.val_use_drift
            scene.hybrid_drift = p.val_drift
            scene.hybrid_auto_pilot = p.val_auto_pilot
            scene.hybrid_break_on_manual = p.val_break_on_manual
    except: pass

def draw_header(self, context):
    if not getattr(context.scene, "hybrid_show_header", True):
        return
        
    layout = self.layout
    scene = context.scene
    layout.separator()
    layout.operator("view3d.hybrid_toggle", text="Kineti-Cam", depress=scene.hybrid_active)

class HYBRID_OT_Toggle(bpy.types.Operator):
    bl_idname = "view3d.hybrid_toggle"
    bl_label = "Toggle Kineti-Cam"
    bl_description = "Start/Stop the Kinetic Engine"

    def execute(self, context):
        context.scene.hybrid_active = not context.scene.hybrid_active
        _state["mode"] = 'MANUAL'
        _state["last_sel_hash"] = 0
        wipe_physics()
        _state["last_shake_offset_loc"] = Vector((0,0,0))
        _state["last_shake_offset_rot"] = Quaternion((1,0,0,0))
        _state["shake_suppressed"] = False
        
        if context.scene.hybrid_active:
            _state["last_loc"] = None
            if not bpy.app.timers.is_registered(update_loop):
                bpy.app.timers.register(update_loop)
        return {'FINISHED'}

class HYBRID_PT_Panel(bpy.types.Panel):
    bl_label = "Geo-Kineti-Cam"
    bl_idname = "VIEW3D_PT_hybrid"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        row = layout.row(align=True)
        row.scale_y = 1.2
        if scene.hybrid_active:
            row.alert = True
            row.operator("view3d.hybrid_toggle", text="Stop Kineti-Cam Engine", icon="PAUSE")
        else:
            row.operator("view3d.hybrid_toggle", text="Start Kineti-Cam Engine", icon="PLAY")
            
        layout.separator()
        layout.prop(scene, "hybrid_show_header", text="Show Header Button")
        
        # --- AUTO-PILOT GROUP ---
        box_auto = layout.box()
        box_auto.prop(scene, "hybrid_auto_pilot", text="Auto-Pilot") 
        box_auto.prop(scene, "hybrid_dist", text="Auto Distance", slider=True)
        box_auto.prop(scene, "hybrid_speed", text="Auto Speed", slider=True)
        
        layout.separator()
        
        # --- MANUAL GROUP (BREAK + DRAG) ---
        box_manual = layout.box()
        box_manual.prop(scene, "hybrid_break_on_manual", text="Break on Manual") 
        box_manual.prop(scene, "hybrid_friction", text="Drag", slider=True)
        
        layout.separator()
        
        # --- SWAY GROUP ---
        box_sway = layout.box()
        col = box_sway.column(align=True)
        col.prop(scene, "hybrid_use_drift", text="Idle Sway") 
        sub = col.column()
        sub.active = scene.hybrid_use_drift 
        sub.prop(scene, "hybrid_drift", text="Intensity", slider=True)
        
        layout.separator()
        
        # --- VERSION FOOTER ---
        row = layout.row()
        row.alignment = 'RIGHT'
        row.enabled = False 
        row.label(text=f"{bl_info['name']} v {bl_info['version'][0]}.{bl_info['version'][1]}.{bl_info['version'][2]}")

classes = (GKC_Preferences, HYBRID_OT_Toggle, HYBRID_PT_Panel)

def register():
    bpy.types.Scene.hybrid_active = bpy.props.BoolProperty(default=False, description="Is the Kineti-Cam engine currently running?")
    bpy.types.Scene.hybrid_show_header = bpy.props.BoolProperty(default=True, description="Show the toggle button in the 3D Viewport header")
    
    bpy.types.Scene.hybrid_auto_pilot = bpy.props.BoolProperty(default=True, update=sync_auto_pilot, description="Automatically fly to and focus on selected mesh elements in Edit Mode")
    bpy.types.Scene.hybrid_break_on_manual = bpy.props.BoolProperty(default=True, update=sync_break, description="Disengage Auto-Pilot instantly when manual navigation is detected")
    bpy.types.Scene.hybrid_dist = bpy.props.FloatProperty(default=3.0, min=0.5, max=5.0, update=sync_dist, description="Distance multiplier when focusing on objects (Auto-Pilot)")
    bpy.types.Scene.hybrid_speed = bpy.props.FloatProperty(default=0.002536, min=0.0, max=3.0, update=sync_speed, description="Interpolation speed for the camera physics and auto-focus")
    bpy.types.Scene.hybrid_friction = bpy.props.FloatProperty(default=0.24474, min=0.0, max=2.0, update=sync_friction, description="Drag/Damping for manual camera movement (Lower = more slide)")
    bpy.types.Scene.hybrid_use_drift = bpy.props.BoolProperty(default=True, update=sync_use_drift, description="Enable idle camera movement")
    bpy.types.Scene.hybrid_drift = bpy.props.FloatProperty(default=0.486902, min=0.0, max=1.0, update=sync_drift, description="Intensity of the idle camera motion (Subtle floating effect)")
    
    for cls in classes: bpy.utils.register_class(cls)
    
    if load_prefs_to_scene not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(load_prefs_to_scene)
    
    bpy.types.VIEW3D_HT_header.append(draw_header)
    
    if bpy.app.timers.is_registered(update_loop):
        bpy.app.timers.unregister(update_loop)

def unregister():
    if bpy.app.timers.is_registered(update_loop): bpy.app.timers.unregister(update_loop)
    if load_prefs_to_scene in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(load_prefs_to_scene)
        
    bpy.types.VIEW3D_HT_header.remove(draw_header)
    
    for cls in classes: bpy.utils.unregister_class(cls)
    del bpy.types.Scene.hybrid_active
    del bpy.types.Scene.hybrid_show_header
    del bpy.types.Scene.hybrid_auto_pilot
    del bpy.types.Scene.hybrid_break_on_manual
    del bpy.types.Scene.hybrid_dist
    del bpy.types.Scene.hybrid_speed
    del bpy.types.Scene.hybrid_friction
    del bpy.types.Scene.hybrid_use_drift
    del bpy.types.Scene.hybrid_drift

if __name__ == "__main__":
    register()