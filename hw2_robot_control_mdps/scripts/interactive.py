import mujoco
import mujoco.viewer
import os
os.environ["MUJOCO_GL"] = "wgl"

from __init__ import *


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)