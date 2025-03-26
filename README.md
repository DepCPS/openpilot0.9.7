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
  --aeb                 enable AEB with compromised data
  --aeb_independent     enable AEB with independent data
  --driver_brake        enable driver brake react
  --driver_lateral      enable driver lateral react
  --ml                  enable ML model
  --scenario [int]      1-6
  --initial_dis [int]   extra relative distance with leading vehicle (m)
  --driver_time [int]   driver reaction time (ms)
```

## MetaDrive

### Launching Metadrive

Start bridge processes, for example, run the simulation with conditon: scenario 1, relative distacne with front vehicle is 60m and driver reaction time is 2.5s.

```
./tools/sim/run_bridge.py --scenario 1 --initial_dis 10 --driver_time 250
```

## Usage examples

1. Determine the type of fault injection and run the fault injection script with corresponding parameter.

    - rd: inject fault into the predicted data of the relative distance to the leading vehicle and causing collision.
    - cur: inject fault into the predicted desired curvature and causing steering.
    - mix: inject both two faults above.
    - none: no fault injection.

2. Choose a scenario nbumber. Default is 1.

    - 1: The lead vehicle cruises at a constant speed (30 mph).
    - 2: The lead vehicle cruises at 30 mph and then accelerates to 40 mph.
    - 3: The lead vehicle cruises at 40 mph and then decelerates to 30 mph.
    - 4: The lead vehicle cruises at 30 mph and suddenly brakes to a stop due to an obstacle.
    - 5: The lead vehicle cruises at 30 mph, and another vehicle from the neighboring lane cuts into the ego vehicle’s driving lane.
    - 6: Two lead vehicles cruise at a constant speed (30 mph) in the same lane; then, the second lead vehicle (the one closer to the ego vehicle) changes lanes and moves into an adjacent lane.

3. Set a extra relative distance with leading vehicle at beginning based on 50m. For example, the default number is 10, so the total relative distance is 60m.

4. Enable safety intervention. Don't add any argument if you don't want to enable any safety intervention.

    - Try each intervention separately.
        - Enable pandas safety checker
        - Enable AEB with cmpromised data
        - Enable AEB with independent data
        - Enable driver reaction
            - brake control
            - steering wheel control
        - Enable ML model
    - Try some combination, such as:
        - Driver, safety checker
        - Driver, safety checker, AEB

5. Additional setting

    - Driver reaction time. Default is 2.5s
        - --driver_time 250
    - Friction.
        - Go to Metadrive component: /openpilot0.9.7/.venv/lib/python3.11/site-packages/metadrive/component/vehicle/base_vehicle.py
        - Find this line
        ```
        wheel_friction = self.config["wheel_friction"] if not self.config["no_wheel_friction"] else 0
        ```
        - Add a multiplier to change the friction. such as:
        ```
        wheel_friction = 0.75 * self.config["wheel_friction"] if not self.config["no_wheel_friction"] else 0
        ```

6. Launch openpilot an d metadrive.

### Single simulation example

1. Initializing fault injection into relative distance.

```
python fault_type.py rd
```

2. Launching openpilot

```
poetry shell
./tools/sim/launch_openpilot.sh
```

3. Launching Metadrive

scenario 1, 60m distance, enable AEB

```
poetry shell
./tools/sim/run_bridge.py --scenario 1 --initial_dis 10 --aeb
```

### Simulation loop

1. Initializing fault injection.

```
python fault_type.py rd
```

2. Run script

```
./simulation_loop.sh
```