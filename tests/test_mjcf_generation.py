import pytest

from handcdo.design_space import DesignSpace
from handcdo.hand_model import build_hand_model
from handcdo.mjcf_generator import build_mjcf_xml
from handcdo.tools import get_tool


def test_mjcf_generation_loadable_when_mujoco_available():
    mujoco = pytest.importorskip("mujoco")
    design = DesignSpace().sample(seed=1)
    xml = build_mjcf_xml(build_hand_model(design), tool=get_tool("hammer"))
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nbody > 1
    assert model.nu > 0
