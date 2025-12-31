Geo-Kineti-Cam is a "drive-by-wire" system for the Blender viewport. Instead of the raw, jerky movement of standard navigation, this addon intercepts your mouse inputs and processes them through a custom physics engine before applying them to the camera.

It intelligently distinguishes between rotation, panning, and zooming to fix common frustrations:

Pivot Anchor: Automatically locks your pivot point in place during rotation, preventing your model from drifting off-screen during fast spins.

Smart Horizon: Enforces a level horizon without restricting you—pitch over the poles and look upside down without the camera rolling or flipping.

Kinetic Coasting: Adds buttery-smooth momentum to all movements, letting you "throw" the camera and watch it glide to a natural stop.

Safety Guards: Automatically detects corrupted navigation data (NaNs) to prevent the camera from breaking or getting lost in deep space.