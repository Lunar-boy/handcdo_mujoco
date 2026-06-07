from handcdo.design_space import DesignSpace, HandDesign, ParameterSpec


def test_sampling_respects_bounds_and_roundtrip(tmp_path):
    space = DesignSpace()
    for seed in range(25):
        design = space.sample(seed=seed)
        validated = space.validate(design.to_dict())
        assert validated == design.to_dict()
    path = tmp_path / "design.json"
    design.to_json(path)
    restored = HandDesign.from_json(path)
    assert restored.design_id == design.design_id
    assert restored.to_dict() == design.to_dict()


def test_hand_design_can_use_explicit_design_space():
    space = DesignSpace((ParameterSpec("toy_length", "float", bounds=(0.0, 1.0)),))

    design = space.sample(seed=0)

    assert set(design.to_dict()) == {"toy_length"}
    assert HandDesign.from_dict(design.to_dict(), space=space).design_id == design.design_id
