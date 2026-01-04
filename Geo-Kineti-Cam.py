bl_info = {
    "name": "Geo-Kineti-Cam",
    "author": "Jiovanie Velazquez",
    "version": (0, 9, 35),
    "blender": (4, 2, 0),
    "location": "View3D > Header & N-Panel",
    "description": "Buttery smooth physics navigation with auto-pilot and organic sway.",
    "category": "View",
}

import bpy
import bmesh
import math
import time
import traceback
from mathutils import Vector, Quaternion, Matrix, Euler

# --- CONSTANTS ---
DEADZONE = 0.0005
ADDON_NAME = __package__ if __package__ else __name__
SCAN_INTERVAL = 0.1 

# --- HELPER FUNCTIONS ---
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
    
    for v in sel_verts:
        w_co = mw @ v.co
        sum_co += w_co
        sum_normal += v.normal
        sel_hash_int += v.index

    center = sum_co / count
    max_dist = 0.0
    for v in sel_verts:
        d = ((mw @ v.co) - center).length
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

# --- THE RIG CLASS ---
class KineticViewRig:
    def __init__(self):
        self.mode = 'MANUAL'
        
        # Physics State
        self.vel_pan = Vector((0,0,0))
        self.vel_zoom = 0.0
        self.vel_rot = Quaternion((1,0,0,0))
        self.is_coasting = False
        self.coasting_start_time = 0.0
        
        # Buffers
        self.buffer_pan = []
        self.buffer_zoom = []
        self.buffer_rot = []
        
        # Tracking
        self.last_loc = None
        self.last_rot = None
        self.last_dist = 0.0
        self.last_is_perspective = True
        self.last_move_time = 0.0
        self.shake_suppressed = False
        
        # Auto-Pilot State
        self.last_scan_time = 0.0
        self.last_sel_hash = 0
        self.target_focus = Vector((0,0,0))
        self.target_dist = 10.0
        self.target_rot = Quaternion((1,0,0,0))
        
        # Drift State
        self.drift_ramp = 0.0 
        self.last_shake_offset_loc = Vector((0,0,0))
        self.last_shake_offset_rot = Quaternion((1,0,0,0))

    def wipe_physics(self):
        self.buffer_pan.clear()
        self.buffer_zoom.clear()
        self.buffer_rot.clear()
        self.vel_pan = Vector((0,0,0))
        self.vel_zoom = 0.0
        self.vel_rot = Quaternion((1,0,0,0))
        self.is_coasting = False

    def tick(self, area, context, prefs):
        rv3d = area.spaces.active.region_3d
        if not rv3d: return

        # 1. Snapshot State
        curr_loc = rv3d.view_location.copy()
        curr_rot = rv3d.view_rotation.copy()
        curr_dist = rv3d.view_distance
        curr_is_persp = rv3d.is_perspective

        if self.last_loc is None:
            self._update_history(curr_loc, curr_rot, curr_dist, curr_is_persp)

        # Calculate Deltas
        diff_loc = curr_loc - self.last_loc
        diff_dist = curr_dist - self.last_dist
        diff_rot = curr_rot @ self.last_rot.inverted()

        speed_loc = diff_loc.length
        speed_rot = diff_rot.angle
        speed_dist = abs(diff_dist)

        mode_switched = (curr_is_persp != self.last_is_perspective)
        view_z = curr_rot.to_matrix().col[2]
        is_axis_aligned = (abs(view_z.x) > 0.9999 or abs(view_z.y) > 0.9999 or abs(view_z.z) > 0.9999)

        if mode_switched or is_axis_aligned:
            self.wipe_physics()
            self.drift_ramp = 0.0 
            self.last_shake_offset_loc = Vector((0,0,0))
            self.last_shake_offset_rot = Quaternion((1,0,0,0))
            self._update_history(curr_loc, curr_rot, curr_dist, curr_is_persp)
            return

        self.last_is_perspective = curr_is_persp
        now = time.time()

        # 2. Auto-Pilot Scanning
        if (now - self.last_scan_time) > SCAN_INTERVAL:
            self._scan_selection(context, prefs, curr_loc, curr_rot)
            self.last_scan_time = now

        is_moving = speed_loc > DEADZONE or speed_dist > DEADZONE or speed_rot > DEADZONE

        # 3. Mode Logic
        if self.mode == 'AUTO':
            self._handle_auto_pilot(rv3d, prefs, curr_loc, curr_rot, curr_dist, is_moving)
        elif self.mode == 'MANUAL':
            self._handle_manual_physics(rv3d, prefs, diff_loc, diff_dist, diff_rot, is_moving, now)

        # 4. Drift / Sway
        self._handle_drift(rv3d, prefs, now, is_moving)

        if area: area.tag_redraw()
        self._update_history(rv3d.view_location.copy(), rv3d.view_rotation.copy(), rv3d.view_distance, curr_is_persp)

    def _update_history(self, loc, rot, dist, persp):
        self.last_loc = loc
        self.last_rot = rot
        self.last_dist = dist
        self.last_is_perspective = persp

    def _scan_selection(self, context, prefs, curr_loc, curr_rot):
        if not prefs.val_auto_pilot or context.mode != 'EDIT_MESH': return

        center, radius, sel_hash, iso_quat = get_target_data_bmesh(context)
        
        if sel_hash and sel_hash != self.last_sel_hash:
            self.last_sel_hash = sel_hash
            self.mode = 'AUTO'
            self.target_focus = center
            self.target_dist = (radius + 0.5) * prefs.val_dist
            
            if iso_quat: 
                self.target_rot = iso_quat
            else:
                direction = center - curr_loc
                if direction.length_squared > 0.001:
                    base_look = direction.to_track_quat('-Z', 'Y')
                    offset = Euler((0.349, 0.349, 0), 'XYZ').to_quaternion()
                    self.target_rot = base_look @ offset
                else: 
                    self.target_rot = curr_rot
            self.wipe_physics()

    def _handle_auto_pilot(self, rv3d, prefs, curr_loc, curr_rot, curr_dist, is_moving):
        if not prefs.val_auto_pilot:
            self.mode = 'MANUAL'
            return

        if prefs.val_break_on_manual and self.last_loc is not None:
            delta_loc_fight = (curr_loc - self.last_loc).length
            delta_rot_fight = curr_rot.rotation_difference(self.last_rot).angle
            tol_loc = 0.01 + (prefs.val_drift * 0.2)
            tol_rot = 0.008 + (prefs.val_drift * 0.1)
            if delta_loc_fight > tol_loc or delta_rot_fight > tol_rot: 
                self.mode = 'MANUAL'
                return

        speed = prefs.val_speed / 10.0
        new_loc = curr_loc.lerp(self.target_focus, speed)
        new_dist = curr_dist + ((self.target_dist - curr_dist) * speed)
        raw_rot = curr_rot.slerp(self.target_rot, speed)
        new_rot = stabilize_horizon(raw_rot)
        
        rv3d.view_location = new_loc
        rv3d.view_distance = new_dist
        rv3d.view_rotation = new_rot

    def _handle_manual_physics(self, rv3d, prefs, diff_loc, diff_dist, diff_rot, is_moving, now):
        if is_moving:
            self.is_coasting = False
            self.shake_suppressed = True
            self.last_move_time = now

            self.buffer_pan.append(diff_loc)
            self.buffer_zoom.append(diff_dist)
            self.buffer_rot.append(diff_rot)
            if len(self.buffer_pan) > 3: 
                del self.buffer_pan[0]
                del self.buffer_zoom[0]
                del self.buffer_rot[0]
        else:
            if not self.is_coasting and self.buffer_pan:
                self.vel_pan = average_vec(self.buffer_pan)
                self.vel_zoom = average_float(self.buffer_zoom)
                self.vel_rot = average_quat(self.buffer_rot)
                
                has_nrg = self.vel_pan.length > 0.001 or abs(self.vel_zoom) > 0.001 or self.vel_rot.angle > 0.0001
                if has_nrg: 
                    self.is_coasting = True
                    self.coasting_start_time = time.time()
                    self.shake_suppressed = False
                    if self.vel_rot.angle > 0.003: self.vel_pan = Vector((0,0,0))
                self.buffer_pan.clear(); self.buffer_zoom.clear(); self.buffer_rot.clear()
        
        # Shake Logic
        if not is_moving and (now - self.last_move_time < 0.3):
            self.shake_suppressed = True
        elif not is_moving and not self.is_coasting:
            self.shake_suppressed = False

        if self.is_coasting:
            self._apply_coasting(rv3d, prefs)

    def _apply_coasting(self, rv3d, prefs):
        friction = 0.98 - (prefs.val_friction * 0.08)
        if friction < 0: friction = 0
        
        self.vel_pan *= friction
        self.vel_zoom *= friction
        identity = Quaternion((1,0,0,0))
        self.vel_rot = self.vel_rot.slerp(identity, 1.0 - friction)
        
        if self.vel_pan.length < 0.001 and abs(self.vel_zoom) < 0.001 and self.vel_rot.angle < 0.0001:
            self.is_coasting = False
        else:
            rv3d.view_location += self.vel_pan
            
            pred_dist = rv3d.view_distance + self.vel_zoom
            if pred_dist < 0.01:
                rv3d.view_distance = 0.01
                self.vel_zoom = 0.0
            else:
                rv3d.view_distance = pred_dist

            new_rot = self.vel_rot @ rv3d.view_rotation
            target_rot = stabilize_horizon(new_rot)
            coast_duration = time.time() - self.coasting_start_time
            stab_alpha = 0.1
            if coast_duration < 0.5: stab_alpha = (coast_duration / 0.5) * 0.1
            rv3d.view_rotation = new_rot.slerp(target_rot, stab_alpha)

    def _handle_drift(self, rv3d, prefs, now, is_moving):
        # Determine Target Strength (0.0 or 1.0)
        target_ramp = 0.0
        
        # KEY CHANGE: Removed "not self.is_coasting"
        # Now we only suppress drift if you are ACTIVELY dragging the mouse (is_moving)
        if prefs.val_use_drift and not is_moving and self.mode == 'MANUAL':
            target_ramp = 1.0
        
        # Smoothly interpolate drift_ramp
        ramp_speed = 0.01 
        self.drift_ramp += (target_ramp - self.drift_ramp) * ramp_speed
        
        # Apply Drift only if ramp is significant
        if self.drift_ramp > 0.001:
            strength = prefs.val_drift * 0.05 * self.drift_ramp 
            rot_strength = strength * 0.2 
            
            n_x = (math.sin(now * 1.2) + math.cos(now * 2.1) * 0.5) * strength
            n_y = (math.cos(now * 1.4) + math.sin(now * 2.4) * 0.5) * strength
            n_z = (math.sin(now * 0.5) * 0.5) * strength
            rot_pitch = math.sin(now * 0.8) * rot_strength
            rot_yaw = math.cos(now * 1.1) * rot_strength
            rot_roll = math.sin(now * 1.6) * (rot_strength * 0.5) 
            
            curr_shake_offset_loc = Vector((n_x, n_y, n_z))
            curr_shake_offset_rot = Euler((rot_pitch, rot_yaw, rot_roll), 'XYZ').to_quaternion()
            
            delta_shake_loc = curr_shake_offset_loc - self.last_shake_offset_loc
            delta_world_loc = rv3d.view_rotation @ delta_shake_loc
            rv3d.view_location += delta_world_loc
            
            delta_shake_rot = curr_shake_offset_rot @ self.last_shake_offset_rot.inverted()
            rv3d.view_rotation = delta_shake_rot @ rv3d.view_rotation
            
            self.last_shake_offset_loc = curr_shake_offset_loc
            self.last_shake_offset_rot = curr_shake_offset_rot
        else:
            # Clean up residual drift if stopped
            if self.last_shake_offset_loc.length_squared > 0:
                 self.last_shake_offset_loc = Vector((0,0,0))
                 self.last_shake_offset_rot = Quaternion((1,0,0,0))

# --- MODAL OPERATOR ---
class GKC_OT_Toggle(bpy.types.Operator):
    bl_idname = "view3d.gkc_toggle"
    bl_label = "Toggle Kineti-Cam"
    bl_description = "Start/Stop the Kinetic Engine"

    _timer = None
    _view_rigs = {}

    def modal(self, context, event):
        try:
            if not getattr(context.scene, "gkc_active", False):
                return self.cancel(context)
        except:
            return self.cancel(context)

        if event.type == 'TIMER':
            try:
                prefs = context.preferences.addons[ADDON_NAME].preferences
                for window in context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D':
                            if area not in self._view_rigs:
                                self._view_rigs[area] = KineticViewRig()
                            self._view_rigs[area].tick(area, context, prefs)
            except Exception:
                return self.cancel(context)

        return {'PASS_THROUGH'}

    def execute(self, context):
        context.scene.gkc_active = not context.scene.gkc_active
        if context.scene.gkc_active:
            self._view_rigs = {}
            self._timer = context.window_manager.event_timer_add(0.01, window=context.window)
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        return {'FINISHED'}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        self._view_rigs = {}
        context.scene.gkc_active = False
        return {'FINISHED'}

# --- PREFS ---
class GKC_Preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_NAME
    
    val_auto_pilot: bpy.props.BoolProperty(default=True, name="Enabled", description="Automatically fly to and focus on selected mesh elements in Edit Mode")
    val_break_on_manual: bpy.props.BoolProperty(default=True, name="Break on Manual", description="Disengage Auto-Pilot instantly when manual navigation is detected")
    val_dist: bpy.props.FloatProperty(default=3.0, min=0.5, max=5.0, name="Distance", description="Distance multiplier when focusing on objects (Auto-Pilot)")
    val_speed: bpy.props.FloatProperty(default=0.002536, min=0.0, max=3.0, name="Speed", description="Interpolation speed for the camera physics and auto-focus")
    val_friction: bpy.props.FloatProperty(default=0.24474, min=0.0, max=2.0, name="Friction (Drag)", description="Drag/Damping for manual camera movement")
    val_use_drift: bpy.props.BoolProperty(default=True, name="Enabled", description="Enable idle camera movement")
    val_drift: bpy.props.FloatProperty(default=0.486902, min=0.0, max=1.0, name="Intensity", description="Intensity of the idle camera motion")

    def draw(self, context):
        layout = self.layout
        layout.label(text="Global Defaults")
        box = layout.box()
        box.label(text="Auto-Pilot")
        box.prop(self, "val_auto_pilot")
        box.prop(self, "val_dist")
        box.prop(self, "val_speed")
        box = layout.box()
        box.label(text="Manual Physics")
        box.prop(self, "val_break_on_manual")
        box.prop(self, "val_friction")
        box = layout.box()
        box.label(text="Idle Sway")
        box.prop(self, "val_use_drift")
        box.prop(self, "val_drift")

# --- UI ---
def draw_header(self, context):
    if not getattr(context.scene, "gkc_show_header", True): return
    layout = self.layout
    layout.operator("view3d.gkc_toggle", text="Kineti-Cam", depress=context.scene.gkc_active)

class GKC_OT_Reset(bpy.types.Operator):
    bl_idname = "view3d.gkc_reset"
    bl_label = "Reset Settings"
    def execute(self, context):
        prefs = context.preferences.addons[ADDON_NAME].preferences
        prefs.property_unset("val_auto_pilot")
        prefs.property_unset("val_dist")
        prefs.property_unset("val_speed")
        prefs.property_unset("val_break_on_manual")
        prefs.property_unset("val_friction")
        prefs.property_unset("val_use_drift")
        prefs.property_unset("val_drift")
        return {'FINISHED'}

class GKC_PT_Panel(bpy.types.Panel):
    bl_label = "Geo-Kineti-Cam"
    bl_idname = "VIEW3D_PT_gkc"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        prefs = context.preferences.addons[ADDON_NAME].preferences
        
        row = layout.row(align=True)
        row.scale_y = 1.2
        if scene.gkc_active:
            row.alert = True
            row.operator("view3d.gkc_toggle", text="Stop Kineti-Cam Engine", icon="PAUSE")
        else:
            row.operator("view3d.gkc_toggle", text="Start Kineti-Cam Engine", icon="PLAY")
            
        layout.separator()
        layout.prop(scene, "gkc_show_header", text="Show Header Button")
        
        # --- AUTO-PILOT GROUP ---
        box_auto = layout.box()
        row = box_auto.row()
        row.label(text="Auto-Pilot")
        row.operator("view3d.gkc_reset", text="", icon="LOOP_BACK")
        
        box_auto.prop(prefs, "val_auto_pilot", text="Enabled") 
        box_auto.prop(prefs, "val_dist", text="Auto Distance", slider=True)
        box_auto.prop(prefs, "val_speed", text="Auto Speed", slider=True)
        
        layout.separator()
        
        # --- MANUAL GROUP ---
        box_manual = layout.box()
        box_manual.prop(prefs, "val_break_on_manual", text="Break on Manual") 
        box_manual.prop(prefs, "val_friction", text="Drag", slider=True)
        
        layout.separator()
        
        # --- SWAY GROUP ---
        box_sway = layout.box()
        col = box_sway.column(align=True)
        col.prop(prefs, "val_use_drift", text="Idle Sway") 
        sub = col.column()
        sub.active = prefs.val_use_drift 
        sub.prop(prefs, "val_drift", text="Intensity", slider=True)

classes = (GKC_Preferences, GKC_OT_Toggle, GKC_PT_Panel, GKC_OT_Reset)

def register():
    bpy.types.Scene.gkc_active = bpy.props.BoolProperty(default=False, description="Is the engine running?", options={'SKIP_SAVE'})
    bpy.types.Scene.gkc_show_header = bpy.props.BoolProperty(default=True, description="Show Header", options={'SKIP_SAVE'})
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.VIEW3D_HT_header.append(draw_header)

def unregister():
    bpy.types.VIEW3D_HT_header.remove(draw_header)
    for cls in classes: bpy.utils.unregister_class(cls)
    del bpy.types.Scene.gkc_active
    del bpy.types.Scene.gkc_show_header

if __name__ == "__main__":
    register()