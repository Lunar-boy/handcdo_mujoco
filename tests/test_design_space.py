from handcdo.design_space import DesignSpace, HandDesign


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
