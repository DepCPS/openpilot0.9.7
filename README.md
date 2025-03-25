# OpenPilot 0.9.7

## System requirements

* Ubuntu 20.04
* Python 3.11

## Set up

1. Clone the code.

```
git clone https://github.com/DepCPS/openpilot0.9.7.git
```

NOTE: This repository uses Git LFS for large files. Ensure you have Git LFS installed and set up before cloning or working with it.

2. Run the setup script

```
cd openpilot0.9.7
git lfs pull
tools/ubuntu_setup.sh

```

Activate a shell with the Python dependencies installed:

```
poetry shell
```

3. Build openpilot

```
scons -u -j$(nproc)
```

## Fault injection

```
python fault_type.py [arguments]
```
- arguments:
    - rd: relative distance
    - cur: curvature
    - mix: both
    - none: no fault

## Launching openpilot

```
./tools/sim/launch_openpilot.sh
```

## Bridge usage

```
$ ./run_bridge.py -h
usage: run_bridge.py [-h] [--joystick] [--high_quality] [--dual_camera]
Bridge between the simulator and openpilot.

options:
  -h, --help            show this help message and exit
  --joystick
  --high_quality
  --dual_camera
  --safety_checker      enable pandas safety checker
  --aeb                 enable AEB
  --driver_brake        enable driver brake react
  --driver_lateral      enable driver lateral react
  --ml                  enable ML model
  --scenario [int]
  --initial_dis [int]
  --driver_time [int]
```

## MetaDrive

### Launching Metadrive

Start bridge processes located in tools/sim:

```
./run_bridge.py --scenario 1 --initial_dis 10 --driver_time 250
```

## Usage examples

### Single simulation

1. Initializing fault injection.

```
python fault_type.py none
```

2. Launching openpilot

```
poetry shell
./tools/sim/launch_openpilot.sh
```

3. Launching Metadrive

```
poetry shell
./tools/sim/run_bridge.py --scenario 1 --initial_dis 10 --driver_time 250
```

### Simulation loop

1. Initializing fault injection.

```
python fault_type.py none
```

2. Run script

```
./simulation_loop.sh
```