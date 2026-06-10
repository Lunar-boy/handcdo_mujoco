# PR 12: ROS 2 Robot Description, Visualization, and Evaluation Bridge

Implement PR 12: add an optional ROS 2 integration layer that exports HandCDO designs as robot descriptions, visualizes them in RViz, and exposes existing design/evaluation/optimization workflows through ROS 2 interfaces.

This PR intentionally combines the earlier planned "URDF exporter", "ROS 2 visualization package", and "ROS 2 service/action wrappers" into one larger but coherent integration PR.

This PR depends on the core CPU MuJoCo pipeline being stable. It does not depend on MuJoCo Warp, but it should remain compatible with the experimental `mujoco_warp` backend introduced in PR 11 when that backend is explicitly requested.

---

## Repository context

This repository is a CPU-only MuJoCo reproduction of the optimization infrastructure from arXiv:2604.27557, with later optional GPU acceleration through MuJoCo Warp.

Existing code already supports:

- `DesignSpace` and `HandDesign`;
- deterministic design generation;
- MJCF generation from hand designs;
- CPU MuJoCo grasp/wrench evaluation;
- backend abstraction through `SimulatorBackend`;
- batch design evaluation;
- result JSON collection into CSV;
- Optuna TPE optimization;
- multi-fidelity evaluation;
- surrogate-assisted candidate proposal;
- optional MuJoCo Warp benchmark/backend from PR 10 and PR 11.

This PR adds a ROS 2 adapter layer. It must not replace the MuJoCo evaluation pipeline.

---

## High-level goal

Add optional ROS 2 support for:

1. exporting generated hand designs to URDF / robot_description;
2. visualizing generated hand morphologies in RViz;
3. publishing design metadata and evaluation results;
4. wrapping existing design generation, evaluation, and optimization entry points as ROS 2 services/actions;
5. preserving a clean separation between the core `handcdo` research package and the optional ROS 2 workspace.

This is still a software-infrastructure PR, not a real-robot control PR.

---

## Non-negotiable architectural boundary

The core `handcdo` package must remain usable without ROS 2.

Default installation and default tests must not require:

- ROS 2;
- `rclpy`;
- `ament`;
- `colcon`;
- RViz;
- MoveIt 2;
- ros2_control;
- Gazebo;
- CUDA;
- MuJoCo Warp;
- real robot hardware.

ROS 2 code must live under an isolated ROS 2 workspace directory:

```text
ros2_ws/src/
```

Do not import `rclpy` from top-level `handcdo` modules.

Do not add ROS 2 dependencies to `[project.dependencies]` in the root `pyproject.toml`.

---

## Required changes: core package

Add pure-Python robot-description export utilities that do not require ROS 2:

```text
handcdo/robot_description/
  __init__.py
  urdf_exporter.py
  joint_mapping.py
  metadata.py
```

Suggested API:

```python
def build_urdf_xml(
    design: HandDesign,
    geometry_config: GeometryConfig | None = None,
    robot_name: str | None = None,
    include_collision: bool = True,
    include_visual: bool = True,
    include_ros2_control: bool = False,
) -> str:
    ...
```

```python
def write_urdf(
    design: HandDesign,
    output_dir: str | Path,
    geometry_config: GeometryConfig | None = None,
    robot_name: str | None = None,
    include_collision: bool = True,
    include_visual: bool = True,
    include_ros2_control: bool = False,
) -> Path:
    ...
```

```python
def hand_design_to_joint_map(design: HandDesign) -> dict[str, float]:
    ...
```

```python
def build_robot_description_metadata(
    design: HandDesign,
    urdf_path: str | Path,
    geometry_config: GeometryConfig | None = None,
) -> dict[str, Any]:
    ...
```

The URDF exporter should:

- use the existing `HandDesign` and `GeometryConfig`;
- reuse existing hand model construction logic where practical;
- produce deterministic link and joint names;
- preserve enough joint limit and geometry metadata to support later MoveIt 2 / ros2_control work;
- write a metadata JSON next to the URDF;
- not attempt to exactly reproduce MJCF physics semantics;
- clearly document that MuJoCo MJCF remains the reference for scoring.

Add a CLI:

```text
scripts/export_hand_urdf.py
```

Suggested usage:

```bash
python3 scripts/export_hand_urdf.py \
  --design-json outputs/designs/<design_id>/design.json \
  --config configs/eval_high.yaml \
  --output-dir outputs/urdf/<design_id> \
  --robot-name handcdo_<design_id>
```

Supported args:

- `--design-json`, required
- `--config`, optional
- `--output-dir`, required
- `--robot-name`, optional
- `--include-collision` / `--no-include-collision`
- `--include-visual` / `--no-include-visual`
- `--include-ros2-control` / `--no-include-ros2-control`, default false

Output:

```text
<output-dir>/robot.urdf
<output-dir>/robot_description_metadata.json
```

---

## Required changes: ROS 2 workspace

Add an optional ROS 2 workspace:

```text
ros2_ws/
  README.md
  src/
    handcdo_msgs/
    handcdo_ros2/
```

### `handcdo_msgs`

Create a ROS 2 interface package:

```text
ros2_ws/src/handcdo_msgs/
  package.xml
  CMakeLists.txt
  msg/
    HandDesign.msg
    GraspCandidate.msg
    HandEvaluation.msg
    OptimizationStatus.msg
  srv/
    GenerateDesign.srv
    ExportRobotDescription.srv
    EvaluateDesign.srv
  action/
    OptimizeHand.action
```

Suggested message contents:

```text
# HandDesign.msg
string design_id
string design_json
string[] parameter_names
float64[] numeric_values
string[] categorical_names
string[] categorical_values
```

```text
# GraspCandidate.msg
string tool
float64 dx
float64 dy
float64 dz
float64 yaw
float64 pitch
float64 roll
float64 closure
float64 thumb_closure
float64 spread_bias
float64 score
bool failed
string error
```

```text
# HandEvaluation.msg
string design_id
float64 hand_score
string backend
bool failed
string error
string result_json
```

```text
# OptimizationStatus.msg
uint32 completed_trials
float64 current_best_score
string current_best_design_id
string status
```

Suggested services:

```text
# GenerateDesign.srv
string search_space_yaml
uint32 seed
---
bool success
string design_id
string design_json
string error
```

```text
# ExportRobotDescription.srv
string design_json
string config_yaml
string output_dir
string robot_name
bool include_collision
bool include_visual
---
bool success
string urdf_path
string robot_description
string metadata_json
string error
```

```text
# EvaluateDesign.srv
string design_json
string config_yaml
string[] tools
string backend
uint32 n_grasp_trials
uint32 seed
string output_dir
---
bool success
float64 hand_score
string result_json
string error
```

Suggested action:

```text
# OptimizeHand.action
string search_space_yaml
string config_yaml
string[] tools
string backend
uint32 n_trials
uint32 n_grasp_trials
uint32 seed
string output_dir
---
bool success
float64 best_score
string best_design_json
string best_result_json
string error
---
uint32 completed_trials
float64 current_best_score
string current_best_design_id
string status
```

Keep the interface schema simple and JSON-friendly. The existing repo should remain the source of truth for full payload semantics.

### `handcdo_ros2`

Create a Python ROS 2 package:

```text
ros2_ws/src/handcdo_ros2/
  package.xml
  setup.py
  setup.cfg
  resource/handcdo_ros2
  handcdo_ros2/
    __init__.py
    converters.py
    design_server.py
    robot_description_server.py
    evaluation_server.py
    optimization_action_server.py
    robot_description_publisher.py
    result_publisher.py
  launch/
    visualize_design.launch.py
    evaluation_bridge.launch.py
  rviz/
    handcdo_visualization.rviz
  test/
    test_import.py
    test_converters.py
```

Implementation requirements:

- ROS 2 nodes should import core `handcdo` modules lazily.
- Nodes should take filesystem paths as parameters instead of assuming repo root.
- Nodes must not require MuJoCo Warp unless backend `mujoco_warp` is explicitly requested.
- ROS 2 services/actions should call existing core functions rather than reimplementing evaluation or optimization logic.
- Node failures should return structured error strings rather than crashing when possible.

---

## ROS 2 nodes

### 1. `robot_description_publisher.py`

Purpose:

- load a `design.json`;
- export URDF;
- publish `robot_description` parameter or topic as appropriate;
- publish `sensor_msgs/JointState` for a neutral/default pose;
- allow RViz visualization through `robot_state_publisher`.

Parameters:

```yaml
design_json: ""
config_yaml: ""
robot_name: "handcdo_hand"
publish_rate: 10.0
include_collision: true
include_visual: true
```

### 2. `design_server.py`

Purpose:

- expose `GenerateDesign.srv`;
- call `DesignSpace.from_yaml(...).sample(seed=...)`;
- return serialized design JSON.

### 3. `robot_description_server.py`

Purpose:

- expose `ExportRobotDescription.srv`;
- call `build_urdf_xml` and/or `write_urdf`;
- return URDF path and robot_description string.

### 4. `evaluation_server.py`

Purpose:

- expose `EvaluateDesign.srv`;
- call existing `evaluate_design` flow;
- support `mujoco`, `mujoco_cpu`, and optionally `mujoco_warp` only when available and requested;
- write result JSON under the requested output directory.

### 5. `optimization_action_server.py`

Purpose:

- expose `OptimizeHand.action`;
- run a bounded optimization using existing Optuna/TPE logic or a thin wrapper around it;
- periodically publish feedback with current best score/design;
- write outputs under requested output directory.

This action server must be conservative: no unbounded optimization loops.

### 6. `result_publisher.py`

Purpose:

- optionally watch or read result JSON/CSV files;
- publish `HandEvaluation` messages for downstream visualization/logging;
- do not become part of the core evaluation loop.

---

## Launch files

Add:

```text
ros2_ws/src/handcdo_ros2/launch/visualize_design.launch.py
ros2_ws/src/handcdo_ros2/launch/evaluation_bridge.launch.py
```

`visualize_design.launch.py` should launch:

- robot description publisher;
- `robot_state_publisher`;
- optionally RViz with `handcdo_visualization.rviz`.

`evaluation_bridge.launch.py` should launch:

- design server;
- robot description server;
- evaluation server;
- optional optimization action server.

Launch files should accept:

- `design_json`;
- `config_yaml`;
- `repo_root`;
- `output_dir`;
- `use_rviz`.

---

## README / documentation

Add:

```text
ros2_ws/README.md
docs/ros2_integration.md
```

Document:

- ROS 2 integration is optional.
- The root `handcdo` package does not require ROS 2.
- Use ROS 2 Jazzy or newer unless the user has a specific distribution requirement.
- Use `colcon build` inside `ros2_ws`.
- Source both the ROS 2 installation and the workspace setup.
- MuJoCo remains the scoring reference.
- RViz visualization is not physical validation.
- Services/actions are wrappers around existing repo functions.

Include examples:

```bash
# Build
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build
source install/setup.bash
```

```bash
# Export URDF without ROS 2
python3 scripts/export_hand_urdf.py \
  --design-json outputs/designs/<design_id>/design.json \
  --config configs/eval_high.yaml \
  --output-dir outputs/urdf/<design_id>
```

```bash
# Visualize in ROS 2
ros2 launch handcdo_ros2 visualize_design.launch.py \
  design_json:=/absolute/path/to/design.json \
  config_yaml:=/absolute/path/to/configs/eval_high.yaml \
  use_rviz:=true
```

```bash
# Start evaluation service bridge
ros2 launch handcdo_ros2 evaluation_bridge.launch.py \
  repo_root:=/absolute/path/to/handcdo_mujoco \
  output_dir:=/absolute/path/to/outputs/ros2_eval
```

---

## Tests

### Root Python tests

Add tests that do not require ROS 2:

```text
tests/test_urdf_exporter.py
tests/test_robot_description_metadata.py
```

Cover:

1. URDF exporter can build XML from a deterministic `HandDesign`.
2. Output XML contains deterministic robot/link/joint names.
3. Joint limits are present where applicable.
4. Metadata JSON contains `design_id`, parameter payload, and URDF path.
5. Export CLI `--help` works.
6. Export CLI writes `robot.urdf` and metadata for a small deterministic design.
7. No ROS 2 imports are required for root tests.

### ROS 2 package tests

Add lightweight tests under `ros2_ws/src/handcdo_ros2/test/`.

Cover:

1. Package imports when ROS 2 is sourced.
2. Converter functions preserve `design_id` and JSON payload.
3. Launch files are syntactically valid if launch testing is available.

Default root `pytest -q` must not require ROS 2. ROS 2 tests should be run separately with `colcon test`.

---

## Validation

Root environment:

```bash
pytest -q
python3 scripts/export_hand_urdf.py --help
python3 scripts/export_hand_urdf.py \
  --design-json outputs/designs/<design_id>/design.json \
  --config configs/eval_fast.yaml \
  --output-dir outputs/urdf_smoke
```

ROS 2 environment:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build
colcon test
source install/setup.bash
ros2 launch handcdo_ros2 visualize_design.launch.py \
  design_json:=/absolute/path/to/design.json \
  config_yaml:=/absolute/path/to/configs/eval_fast.yaml \
  use_rviz:=false
```

Do not commit generated outputs, build/install/log directories, RViz user state, caches, or virtual environments.

---

## Out of scope

Do not implement:

- MoveIt 2 planning configuration;
- ros2_control controllers;
- real hardware drivers;
- UR5e integration;
- tactile sensors;
- OptiTrack integration;
- Gazebo as a scoring backend;
- replacement of MuJoCo scoring;
- changes to design-space bounds;
- changes to CPU MuJoCo semantics;
- changes to MuJoCo Warp backend semantics;
- fabrication workflow.

MoveIt 2 and ros2_control belong to PR 13.

---

## Success criteria

This PR is successful if:

1. The root repo remains usable and testable without ROS 2.
2. A design JSON can be exported to URDF deterministically.
3. A ROS 2 workspace exists under `ros2_ws/src`.
4. RViz visualization can be launched from a generated design in a ROS 2 environment.
5. Generate/evaluate/optimize workflows are exposed through ROS 2 interfaces without reimplementing the core algorithms.
6. MuJoCo remains the reference scoring backend.
