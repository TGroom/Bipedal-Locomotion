
import math
import bpy
from mathutils import Euler

ctx = bpy.context.copy()

bpy.ops.render.render('INVOKE_DEFAULT')

###This is handled with some handlers, but essentially seems to work with a simple context copy

for s in bpy.data.scenes:
     if s == "CreatingFrames":
         ctx['window'].scene = s

bpy.context.window.scene = bpy.data.scenes.get("CreatingFrames")

import os
blend_path = bpy.data.filepath
#assert blend_path # abort if .blend is not unsaved
blend_path = os.path.dirname(bpy.path.abspath(blend_path))

C = bpy.context
Body = C.scene.objects.get("Body")
hipRight = C.scene.objects.get("hipRight")
hipLeft = C.scene.objects.get("hipLeft")
upperRight = C.scene.objects.get("upperRight")
upperLeft = C.scene.objects.get("upperLeft")
lowerRight = C.scene.objects.get("lowerRight")
lowerLeft = C.scene.objects.get("lowerLeft")
footRight = C.scene.objects.get("footRight")
footLeft = C.scene.objects.get("footLeft")

hipRight.rotation_euler = Euler((0, 0, 0), 'XYZ')
hipLeft.rotation_euler = Euler((0, 0, 0), 'XYZ')
upperRight.rotation_euler = Euler((0, 0, 0), 'XYZ')
upperLeft.rotation_euler = Euler((0, 0, 0), 'XYZ')
lowerRight.rotation_euler = Euler((0, 0, 0), 'XYZ')
lowerLeft.rotation_euler = Euler((0, 0, 0), 'XYZ')
footRight.rotation_euler = Euler((0, 0, 0), 'XYZ')
footLeft.rotation_euler = Euler((0, 0, 0), 'XYZ')

"""
Copy and paste PCA loadings into the following arrays
Run the code using the play button above
The output images will be saved in the same directory
"""

# Walking FINAL
PCA1 = [-0.19704756262851947, -0.11042972455114675, -0.5341333565042145, 0.5268530057629807, 0.44186109235579474, -0.43687959078894445, ]
PCA2 = [0.011889496904311203, -0.5144474470712127, 0.2807566692058967, 0.24254407111335374, -0.5792205768234787, -0.5119121542938028, ]
PCA3 = [-0.02293020907449269, -0.8386387517335533, -0.015788787475522592, -0.09900621020270212, 0.3100767310756904, 0.43584418658432716, ]

# Stair Traversal FINAL
#PCA1 = [-0.15242284205485962, -0.018470046153743147, -0.5823604964707402, 0.4089552375669817, 0.36350797269265217, -0.5812916257961775, ]
#PCA2 = [-0.1301301752041975, -0.4688970064266461, 0.19191476865867702, 0.5679749018077054, -0.6215043560674258, -0.13231516444415325, ]
#PCA3 = [0.24330727214705264, -0.8455114979250241, -0.273023616960849, -0.3139590835798885, 0.18743272926111515, 0.13292346902621655, ]


PCA = [PCA1, PCA2, PCA3]

PCAnum = 1  # Change this number to 1, 2 or 3 to specify which PCA to visualise

import bpy
import random

def hyperspherical_to_cartesian(polar):
    r, theta1, theta2, theta3, theta4, theta5 = polar
    x1 = r * math.cos(theta1)
    x2 = r * math.sin(theta1) * math.cos(theta2)
    x3 = r * math.sin(theta1) * math.sin(theta2) * math.cos(theta3)
    x4 = r * math.sin(theta1) * math.sin(theta2) * math.sin(theta3) * math.cos(theta4)
    x5 = r * math.sin(theta1) * math.sin(theta2) * math.sin(theta3) * math.sin(theta4) * math.cos(theta5)
    x6 = r * math.sin(theta1) * math.sin(theta2) * math.sin(theta3) * math.sin(theta4) * math.sin(theta5)
    return [x1,x2,x3,x4,x5,x6]

class ModalTimerOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"

    limits : bpy.props.IntProperty(default=0) #not 'limits ='
    _timer = None
    scale = 0.033333
    frame = 1
    step = 1

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'} or self.limits > 120:
            self.limits = 0
            self.cancel(context)
            
            bpy.ops.render.render(animation=True, write_still=True, scene="CreatingFrames")

            for s in bpy.data.scenes:
                if s == "StackingImages":
                    ctx['window'].scene = s
            bpy.context.window.scene = bpy.data.scenes.get("StackingImages")
    
            for image in bpy.data.images:
                image.reload()
                
            bpy.context.scene.render.filepath = os.path.join(blend_path, 'PCA' + str(PCAnum) + '.png')
            bpy.ops.render.render(write_still = True, scene="StackingImages")

            for s in bpy.data.scenes:
                if s == "CreatingFrames":
                    ctx['window'].scene = s
            bpy.context.window.scene = bpy.data.scenes.get("CreatingFrames")
            
            return {'FINISHED'}

        if event.type == 'TIMER':
            #XYZcoord = (random.random()*100, random.random()*100, random.random()*100)
            #bpy.ops.mesh.primitive_uv_sphere_add(location=XYZcoord)

            hipRight.keyframe_insert(data_path="rotation_euler", frame=self.frame)
            hipLeft.keyframe_insert(data_path="rotation_euler", frame=self.frame)
            upperRight.keyframe_insert(data_path="rotation_euler", frame=self.frame)
            upperLeft.keyframe_insert(data_path="rotation_euler", frame=self.frame)
            lowerRight.keyframe_insert(data_path="rotation_euler", frame=self.frame)
            lowerLeft.keyframe_insert(data_path="rotation_euler", frame=self.frame)
            #footRight.keyframe_insert(data_path="rotation_euler", frame=self.frame)
            #footLeft.keyframe_insert(data_path="rotation_euler", frame=self.frame)
            
            hipRight.rotation_euler = Euler((hipRight.rotation_euler.x, PCA[PCAnum-1][0]*self.scale*self.step, hipRight.rotation_euler.z), 'XYZ')
            hipLeft.rotation_euler = Euler((hipLeft.rotation_euler.x, PCA[PCAnum-1][1]*self.scale*self.step, hipLeft.rotation_euler.z), 'XYZ')
            upperRight.rotation_euler = Euler((PCA[PCAnum-1][2]*self.scale*self.step, upperRight.rotation_euler.y, upperRight.rotation_euler.z), 'XYZ')
            upperLeft.rotation_euler = Euler((PCA[PCAnum-1][3]*self.scale*self.step, upperLeft.rotation_euler.y, upperLeft.rotation_euler.z), 'XYZ')
            lowerRight.rotation_euler = Euler((max(PCA[PCAnum-1][4]*self.scale*self.step,0), lowerRight.rotation_euler.y, lowerRight.rotation_euler.z), 'XYZ')
            lowerLeft.rotation_euler = Euler((max(PCA[PCAnum-1][5]*self.scale*self.step,0), lowerLeft.rotation_euler.y, lowerLeft.rotation_euler.z), 'XYZ')
            #footRight.rotation_euler = Euler((PCA[PCAnum-1][6]*self.scale*self.step, footRight.rotation_euler.y, footRight.rotation_euler.z), 'XYZ')
            #footLeft.rotation_euler = Euler((PCA[PCAnum-1][8]*self.scale*self.step, footLeft.rotation_euler.y, footLeft.rotation_euler.z), 'XYZ')
            
            self.limits += 1
            self.frame += 1
            
            if(self.frame >= 30 and self.frame < 90):
                self.step -= 1
            else:
                self.step += 1
            
        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(time_step=0.002, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)


def register():
    bpy.utils.register_class(ModalTimerOperator)


def unregister():
    bpy.utils.unregister_class(ModalTimerOperator)


if __name__ == "__main__":
    register()
    bpy.ops.wm.modal_timer_operator()
    