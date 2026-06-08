# PR 13: MoveIt 2 and ros2_control Integration Scaffold

Implement PR 13: add an optional MoveIt 2 and ros2_control integration scaffold for generated HandCDO hand descriptions.

This PR intentionally combines the previously separate "MoveIt 2 planning config", "ros2_control mock hardware", and "trajectory/control interface" work into one integration PR.

This PR depends on PR 12. Do not start this PR until the repository can export a generated `HandDesign` to URDF / robot_description and visualize it through the optional ROS 2 workspace.

---

## Repository context

The core repository implements a MuJoCo-based research pipeline for dexterous hand co-design optimization. ROS 2 support from PR 12 provides:

- pure-Python URDF export from `HandDesign`;
- a ROS 2 workspace under `ros2_ws/src`;
- `handcdo_msgs`;
- `handcdo_ros2`;
- RViz visualization;
- ROS 2 services/actions wrapping generation, evaluation, and optimization.

This PR adds optional planning/control scaffolding around those generated robot descriptions.

It must not replace the MuJoCo scoring pipeline.

---

## High-level goal

Add a safe, optional integration layer for:

1. generating MoveIt 2-compatible configuration for a HandCDO hand;
2. defining planning groups for the generated hand;
3. adding ros2_control-compatible URDF tags for mock/simulated position control;
4. launching MoveIt 2 and ros2_control in a read-only or mock-control setup;
5. converting optimized/evaluated grasp candidates into planned joint targets or simple trajectories;
6. preparing the repo for future real hardware work without adding any real hardware driver yet.

This PR is still a scaffold. It is not a real robot deployment PR.

---

## Safety and scope boundary

This PR must be safe by default.

Default behavior must be:

- no real hardware communication;
- no physical robot actuation;
- no UR5e integration;
- no gripper power commands;
- no networked hardware controller;
- no movement unless explicitly using mock/simulated controllers;
- no assumption that generated hands are physically buildable.

MoveIt 2 and ros2_control code must stay inside the optional ROS 2 workspace.

The root `handcdo` package must remain usable without ROS 2, MoveIt 2, or ros2_control.

---

## Required changes: core robot description exporter

Extend the robot description exporter from PR 12 to optionally emit ros2_control tags and MoveIt metadata.

Update:

```text
handcdo/robot_description/urdf_exporter.py
handcdo/robot_description/joint_mapping.py
handcdo/robot_description/metadata.py
```

Add if helpful:

```text
handcdo/robot_description/moveit_metadata.py
```

Required additions:

```python
def build_ros2_control_xml_block(
    design: HandDesign,
    hardware_plugin: str = "mock_components/GenericSystem",
    command_interface: str = "position",
    state_interfaces: tuple[str, ...] = ("position", "velocity"),
) -> str:
    ...
```

```python
def build_moveit_group_metadata(
    design: HandDesign,
) -> dict[str, Any]:
    ...
```

URDF export should support:

```bash
--include-ros2-control
```

and should add deterministic ros2_control tags when requested.

The ros2_control block should use mock/simulated hardware by default.

---

## Required changes: ROS 2 workspace packages

Add or update packages under:

```text
ros2_ws/src/
```

### Update `handcdo_ros2`

Add:

```text
ros2_ws/src/handcdo_ros2/handcdo_ros2/
  trajectory_exporter.py
  joint_target_server.py
  grasp_to_joint_target.py
  safety.py
```

Add launch files:

```text
ros2_ws/src/handcdo_ros2/launch/
  handcdo_moveit_demo.launch.py
  handcdo_ros2_control_mock.launch.py
  handcdo_planning_scene.launch.py
```

Add configs:

```text
ros2_ws/src/handcdo_ros2/config/
  controllers.yaml
  joint_limits.yaml
  kinematics.yaml
  moveit_planning.yaml
  ompl_planning.yaml
```

### Optional new package: `handcdo_moveit_config`

If it is cleaner to follow MoveIt 2 conventions, create:

```text
ros2_ws/src/handcdo_moveit_config/
  package.xml
  CMakeLists.txt
  config/
    handcdo.srdf
    joint_limits.yaml
    kinematics.yaml
    ompl_planning.yaml
    controllers.yaml
    ros2_controllers.yaml
  launch/
    demo.launch.py
    move_group.launch.py
    rviz.launch.py
    spawn_controllers.launch.py
  rviz/
    moveit.rviz
```

This package may be generated from templates, but generated files should be deterministic and reviewable.

Prefer a minimal, static template first. Do not implement a complex auto-generated MoveIt setup assistant replacement unless necessary.

---

## MoveIt 2 configuration requirements

Add enough configuration to load the generated HandCDO hand in MoveIt 2.

Planning groups:

- `hand`;
- one group per finger if deterministic finger naming allows it;
- optionally `thumb`;
- optionally `fingers`.

SRDF should include:

- virtual joint or fixed base definition if needed;
- planning groups;
- end-effector/group metadata if useful;
- disabled collision pairs only when deterministically known and documented.

Joint limits:

- derive from exported design / hand model where possible;
- never exceed MuJoCo model limits silently;
- write `joint_limits.yaml`.

Kinematics:

- use a conservative plugin/config that works for simple joint-space planning;
- if no meaningful IK solver is available for the generated hand, document that only joint-space planning is supported.

Planning:

- configure OMPL for simple joint-space planning;
- no task-level grasp planner is required in this PR.

Launch:

- a demo launch should load robot_description, robot_description_semantic, MoveIt planning config, and RViz;
- it should work with mock controllers and no real hardware.

---

## ros2_control requirements

Add a mock ros2_control setup.

Required:

```text
controllers.yaml
ros2_controllers.yaml
```

At minimum, support:

- `joint_state_broadcaster`;
- a position trajectory controller or equivalent mock position controller for hand joints.

The ros2_control URDF block should define:

- one command interface per actuated joint, default `position`;
- state interfaces, at least `position`;
- velocity state interface if practical;
- deterministic joint names matching URDF and MoveIt config.

Do not add real hardware plugins.

Do not add vendor-specific drivers.

Do not command any real robot.

---

## Joint target and trajectory bridge

Add a conservative bridge from HandCDO outputs to ROS 2 / MoveIt-compatible joint targets.

Suggested functionality:

```python
def grasp_params_to_joint_target(
    design: HandDesign,
    grasp: GraspParams,
) -> dict[str, float]:
    ...
```

```python
def hand_design_neutral_joint_target(
    design: HandDesign,
) -> dict[str, float]:
    ...
```

```python
def write_joint_trajectory_yaml(
    joint_target: dict[str, float],
    output_path: str | Path,
    duration_sec: float = 2.0,
) -> Path:
    ...
```

Important:

- This bridge is not a physically validated grasp execution policy.
- It only maps simple closure/spread-style parameters to joint targets.
- It must clamp joint targets to exported joint limits.
- It must write metadata indicating the source design, grasp, and clamping behavior.

Add CLI:

```text
scripts/export_grasp_joint_target.py
```

Example:

```bash
python3 scripts/export_grasp_joint_target.py \
  --design-json outputs/designs/<design_id>/design.json \
  --grasp-json outputs/results/<design_id>.json \
  --output-yaml outputs/ros2_targets/<design_id>_target.yaml
```

---

## Safety gates

Add explicit safety utilities:

```text
ros2_ws/src/handcdo_ros2/handcdo_ros2/safety.py
```

Requirements:

- default launch files must use mock controllers;
- any launch argument that could imply real hardware must default to false;
- if `use_fake_hardware:=false` or equivalent is passed, the launch should fail unless a separate explicit `allow_real_hardware:=true` flag is provided;
- the README must warn that this repo does not provide certified hardware control.

This may feel conservative, but it prevents accidental misuse.

---

## README / documentation

Add:

```text
docs/moveit2_ros2_control.md
```

Update:

```text
ros2_ws/README.md
```

Document:

- this is optional;
- the root package does not need MoveIt 2 or ros2_control;
- MoveIt 2 is used for planning/visualization scaffold, not for scoring;
- ros2_control uses mock hardware by default;
- no real hardware driver is included;
- generated hands may not be physically buildable;
- final hand quality should still be validated with MuJoCo scores and, later, real experiments.

Include examples:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build
source install/setup.bash
```

```bash
# Generate URDF with ros2_control tags
python3 scripts/export_hand_urdf.py \
  --design-json outputs/designs/<design_id>/design.json \
  --config configs/eval_high.yaml \
  --output-dir outputs/urdf_moveit/<design_id> \
  --include-ros2-control
```

```bash
# Launch mock ros2_control setup
ros2 launch handcdo_ros2 handcdo_ros2_control_mock.launch.py \
  design_json:=/absolute/path/to/design.json \
  config_yaml:=/absolute/path/to/configs/eval_high.yaml
```

```bash
# Launch MoveIt 2 demo
ros2 launch handcdo_ros2 handcdo_moveit_demo.launch.py \
  design_json:=/absolute/path/to/design.json \
  config_yaml:=/absolute/path/to/configs/eval_high.yaml \
  use_fake_hardware:=true
```

Do not claim real grasp execution.

---

## Tests

### Root tests without ROS 2

Add tests that do not require ROS 2:

```text
tests/test_ros2_control_urdf_export.py
tests/test_grasp_joint_target_export.py
```

Cover:

1. `--include-ros2-control` adds a ros2_control XML block.
2. ros2_control joint names match exported URDF joint names.
3. command/state interfaces are present.
4. joint target conversion clamps values to limits.
5. trajectory YAML output is deterministic.
6. CLI `scripts/export_grasp_joint_target.py --help` works.
7. Root tests do not import `rclpy`, MoveIt 2, or ros2_control.

### ROS 2 tests

Add tests under ROS 2 workspace:

```text
ros2_ws/src/handcdo_ros2/test/
  test_moveit_launch_files.py
  test_ros2_control_configs.py
```

Cover where practical:

1. config YAML files parse;
2. launch files are syntactically valid;
3. mock controller config contains `joint_state_broadcaster`;
4. no launch file defaults to real hardware;
5. safety flags reject accidental real-hardware mode.

ROS 2 tests should be run through `colcon test`, not the root pytest suite.

---

## Validation

Root environment:

```bash
pytest -q
python3 scripts/export_hand_urdf.py --help
python3 scripts/export_grasp_joint_target.py --help
python3 scripts/export_hand_urdf.py \
  --design-json outputs/designs/<design_id>/design.json \
  --config configs/eval_fast.yaml \
  --output-dir outputs/urdf_control_smoke \
  --include-ros2-control
```

ROS 2 environment:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build
colcon test
source install/setup.bash
ros2 launch handcdo_ros2 handcdo_ros2_control_mock.launch.py \
  design_json:=/absolute/path/to/design.json \
  config_yaml:=/absolute/path/to/configs/eval_fast.yaml
```

MoveIt 2 environment:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build
source install/setup.bash
ros2 launch handcdo_ros2 handcdo_moveit_demo.launch.py \
  design_json:=/absolute/path/to/design.json \
  config_yaml:=/absolute/path/to/configs/eval_fast.yaml \
  use_fake_hardware:=true
```

Do not commit:

- generated URDF outputs;
- generated MoveIt configs if they are produced under `outputs/`;
- ROS 2 `build/`, `install/`, or `log/` directories;
- RViz runtime state;
- controller logs;
- caches;
- virtual environments.

---

## Out of scope

Do not implement:

- real hardware drivers;
- UR5e integration;
- physical hand actuation;
- tactile sensors;
- OptiTrack;
- camera perception;
- online closed-loop grasp execution;
- Gazebo scoring backend;
- replacement of MuJoCo scoring;
- changes to MuJoCo CPU or Warp score semantics;
- fabrication workflow;
- safety certification.

These are future hardware-deployment projects, not part of this scaffold.

---

## Success criteria

This PR is successful if:

1. The root repo remains testable without ROS 2, MoveIt 2, or ros2_control.
2. URDF export can optionally include ros2_control tags.
3. A MoveIt 2 demo launch can load the generated hand description in a ROS 2 environment.
4. Mock ros2_control controllers can be launched without real hardware.
5. Joint target/trajectory export works deterministically from HandCDO design/grasp payloads.
6. Documentation clearly states that MoveIt 2 and ros2_control are optional deployment scaffolds, not scoring backends.
