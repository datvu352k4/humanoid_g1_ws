import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.URDF(
        file="g1_description/g1_29dof_mode_11_g.urdf",
        pos=(0, 0, 0.78),
        euler=(0, 0, 90),
        scale=1.0,
    ),
)
scene.build()

while True:
    scene.step()
