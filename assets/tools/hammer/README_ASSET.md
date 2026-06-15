# Hammer asset for handcdo_mujoco

Generated from Gazebo Fuel `Cole_Hardware_Hammer_Black`.

Files expected by the repo:

- `visual.obj`: detailed non-colliding visual mesh
- `collision_0.obj`: convex handle-like collision hull
- `collision_1.obj`: convex head-like collision hull
- `texture.png`: original texture, kept for reference; current MJCF generator does not attach it as a MuJoCo material

Transform applied to raw OBJ vertices: translation `[0.01, 0.0086, -0.02]` meters.

Original mesh: 4142 vertices, 7836 faces after transform/export.
Collision hulls:
- `collision_0.obj`: 130 vertices, 256 faces, extents=[0.22628199999999998, 0.023145, 0.076378]
- `collision_1.obj`: 826 vertices, 1648 faces, extents=[0.09527, 0.060032, 0.096066]

License note: source model declares Creative Commons Attribution 4.0 in `model.sdf` / `model.config`.
