#!/usr/bin/env python
import argparse

from typing import Any
from multiprocessing import Queue

from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.bridge.metadrive import metadrive_process

def create_bridge(dual_camera, high_quality):
  queue: Any = Queue()

  simulator_bridge = MetaDriveBridge(dual_camera, high_quality)
  simulator_process = simulator_bridge.run(queue)

  return queue, simulator_process, simulator_bridge

def main():
  _, simulator_process, _ = create_bridge(True, False)
  simulator_process.join()

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between the simulator and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')

  parser.add_argument('--scenario', type=int, default=1)
  parser.add_argument('--initial_dis', type=int, default=50)
  parser.add_argument('--driver_time', type=int, default=250)
  parser.add_argument('--safety_checker', action='store_true', help='Enable pandas safety check')
  parser.add_argument('--aeb', action='store_true', help='Enable AEB')
  parser.add_argument('--driver_brake', action='store_true', help='Enable driver brake react')
  parser.add_argument('--driver_lateral', action='store_true', help='Enable driver lateral react')
  parser.add_argument('--ml', action='store_true', help='Enable ML model')


  return parser.parse_args(add_args)

if __name__ == "__main__":
  args = parse_args()
  metadrive_process.scenario = args.scenario
  metadrive_process.initial_dis = args.initial_dis
  metadrive_process.driver_reaction_time = args.driver_time
  metadrive_process.Panda_SafetyCheck_Enable = args.safety_checker
  metadrive_process.AEB_React_Enable = args.aeb
  metadrive_process.Driver_react_Enable = args.driver_brake
  metadrive_process.Driver_lateral_react_Enable = args.driver_lateral
  metadrive_process.ml_model_Enable = args.ml

  queue, simulator_process, simulator_bridge = create_bridge(args.dual_camera, args.high_quality)

  if args.joystick:
    # start input poll for joystick
    from openpilot.tools.sim.lib.manual_ctrl import wheel_poll_thread

    wheel_poll_thread(queue)
  else:
    # start input poll for keyboard
    from openpilot.tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

    keyboard_poll_thread(queue)

  simulator_bridge.shutdown()

  simulator_process.join()
