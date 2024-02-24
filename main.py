import brax
import jax
import mediapy as media
import mujoco
from dm_control import mjcf
from mujoco import viewer

model = mjcf.RootElement()
assert model.worldbody is not None


model.worldbody.add("geom", name="ground", type="plane", size=[40, 40, 1])

sphere_body = model.worldbody.add("body", name="sphere", pos=[0, 0, 2])
sphere_body.add("joint", name="j", type="free")
sphere_body.add("geom", name="sphere", type="sphere", size=[0.5], mass=1)

model.worldbody.add("light", pos=[0, 0, 3])
model.worldbody.add(
    "camera", name="rgb", pos=[0, 0, 10], fovy=50, mode="targetbody", target=sphere_body
)


mj_model = mujoco.MjModel.from_xml_string(model.to_xml_string())
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

viewer.launch(mj_model)
