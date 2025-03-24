import math
import time
import numpy as np

from collections import namedtuple
from panda3d.core import Vec3
from multiprocessing.connection import Connection

from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation

from openpilot.common.realtime import Ratekeeper

from openpilot.tools.sim.lib.common import vec3
from openpilot.tools.sim.lib.camerad import W, H

from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.policy.idm_policy import IDMPolicy, ManualControllableIDMPolicy
from metadrive.policy.lange_change_policy import LaneChangePolicy
from metadrive.manager import BaseManager
from metadrive.obs.state_obs import LidarStateObservation
from cereal import messaging
from openpilot.common.numpy_fast import clip
import pandas as pd

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.optimizers import Adam
import keras.backend as K

sm = messaging.SubMaster(['radarState', 'modelV2', 'controlsState','carParams'])




# **************************** Global value ************************
openpilot_engaged = False
# Set scenario number
scenario = 0
# Leading car initail distance
initial_dis = 50
# Set whether cut in and move out



lane_change_start_frame = 0

# AEB independent
aeb_independent = False

frameIdx = 0
vEgo = 60 #mph #set in selfdrive/controls/controlsd


Panda_SafetyCheck_Enable = False
Driver_react_Enable = False
Driver_lateral_react_Enable = False
driver_alerted_time = -1
driver_lateral_alerted_time = -1
AEB_React_Enable = False
ml_model_Enable = False
st_steer = 0
st_gas = 0

recovery_mode_gas = False
recovery_mode_steer = False
buffer = []
old_gas = 0
old_steer = 0
ml_gas_action = False
ml_steer_action = False

#if ml_model_Enable:
#  model = load_model('lstm_autonomous_model.h5', compile=False)
#  model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

driver_reaction_time = 250

simulation_done = False


aeb_alert = False
aeb_brake = 0
fcw_alert = False

out_of_lane_alert = False
out_of_lane_count = 0
out_of_lane_count_max = 0

driver_brake_action = False
driver_lateral_action = False

Panda_SafetyCheck_alert = False

rk = Ratekeeper(100, None)
vehicle1 = None
vehicle2 = None
speed_change = 0

lead_rd_gt = 1000
v1_rel_dis_t = 1000
v2_rel_dis_t = 1000

OP_alertTime = []
OP_alertType_list =[]
OP_alertText1_list = []
OP_alertText2_list = []

hazard = False
hazard_time = []
hazard_type = [] # 1: collision; 2: out of lane

AEB_trigger_time = -1
Driver_brake_trigger_time = -1
Driver_lateral_trigger_time = -1
FCW_alert_time = -1
out_of_lane_alert_time = -1



def driver_brake_simulator(t_brake):
  driver_brake_tmp = math.exp(10*t_brake*0.01-12) #1 iteration =0.01 sencods
  driver_brake_out = clip(driver_brake_tmp/(1+driver_brake_tmp),0,1) #1
  #print("driver brake: ", driver_brake_out)
  return driver_brake_out


#IDMPolicy.NORMAL_SPEED = 50
#IDMPolicy.MAX_SPEED = 120
#**************************** Traffic Manager **********************
class MovingExampleManager(BaseManager):

    def before_step(self):
        global vehicle1, vehicle2
        # Scenario 1: slow constant speed 48.28 km/h (30 mph)
        # Scenario 2: high constant speed 64.37 km/h (40 mph)
        if scenario == 1 or scenario == 5 or scenario == 6:
          if openpilot_engaged:
            IDMPolicy.NORMAL_SPEED = 48.28
            IDMPolicy.MAX_SPEED = 48.28
          else:
            IDMPolicy.NORMAL_SPEED = 1
            IDMPolicy.MAX_SPEED = 1

          p1 = self.get_policy(vehicle1.id)
          vehicle1.before_step(p1.act(vehicle1.id))

          if scenario == 5 or scenario == 6:
            global lane_change_start_frame
            #IDMPolicy.TIME_WANTED = 0.5
            IDMPolicy.DISTANCE_WANTED = 5.0
            p2 = self.get_policy(vehicle2.id)
            if lead_rd_gt < 35 and lane_change_start_frame == 0 and frameIdx > 500:
              lane_change_start_frame = frameIdx

            if frameIdx - lane_change_start_frame <= 60 and frameIdx > 500:
              if scenario == 5:
                vehicle2.before_step([0.25,0])
              elif scenario == 6:
                vehicle2.before_step([-0.25,0])
            else:
              if scenario == 6 and frameIdx > 500 and frameIdx < 560:
                vehicle2.before_step([0.2,0])
              elif scenario == 6 and frameIdx > 600 and frameIdx < 650:
                vehicle2.before_step([0,1])
              else:
                vehicle2.before_step(p2.act(vehicle2.id))
        # Scenario 3: Leading car suddenly accelerate
        # Scenario 4: Leading car suddenly brake
        elif scenario == 2 or scenario == 3 or scenario == 4:
          global speed_change
          if (scenario == 2 or scenario == 4) and openpilot_engaged and speed_change == 0:
            IDMPolicy.NORMAL_SPEED = 48.28
            IDMPolicy.MAX_SPEED = 48.28
          elif scenario == 3 and openpilot_engaged and speed_change == 0:
            IDMPolicy.NORMAL_SPEED = 64.37
            IDMPolicy.MAX_SPEED = 64.37
          else:
            IDMPolicy.NORMAL_SPEED = 1
            IDMPolicy.MAX_SPEED = 1

          if lead_rd_gt < 35 and frameIdx > 2000:
            speed_change = 1

          if speed_change == 1:
            if scenario == 2:
              IDMPolicy.NORMAL_SPEED = 64.37
              IDMPolicy.MAX_SPEED = 64.37
            elif scenario == 3:
              IDMPolicy.NORMAL_SPEED = 48.28
              IDMPolicy.MAX_SPEED = 48.28
            elif scenario == 4:
              IDMPolicy.NORMAL_SPEED = 1
              IDMPolicy.MAX_SPEED = 1

          p1 = self.get_policy(vehicle1.id)
          vehicle1.before_step(p1.act(vehicle1.id))



    def reset(self):
        global vehicle1, vehicle2
        # add vehicle1
        if scenario == 6:
          point_x = initial_dis + 10
        else:
          point_x = initial_dis
        if scenario in (1,2,3,4,5,6):
          vehicle1 = self.spawn_object(DefaultVehicle,
                          vehicle_config=dict(),
                          position=(point_x, 9), # position=(point_x, 9),
                          heading=0)
          self.add_policy(vehicle1.id,  ManualControllableIDMPolicy, vehicle1, self.generate_seed())
        # add vehicle2
        if scenario == 5 or scenario == 6:
          if scenario == 5:
            initial_lateral = 4
          elif scenario == 6:
            initial_lateral = 9
          vehicle2 = self.spawn_object(DefaultVehicle,
                          vehicle_config=dict(),
                          position=(point_x - 10, 4),
                          heading=0)
          self.add_policy(vehicle2.id,  ManualControllableIDMPolicy, vehicle2, self.generate_seed())

    def after_step(self):
        for obj in self.spawned_objects.values():
            obj.after_step()
        #if self.episode_step == 180:
            #self.clear_objects(list(self.spawned_objects.keys()))

class MovingExampleEnv(MetaDriveEnv):

    def setup_engine(self):
        super(MovingExampleEnv, self).setup_engine()
        self.engine.update_manager("traffic_manager", MovingExampleManager()) # replace existing traffic manager
# **************************** end ****************************

C3_POSITION = Vec3(0.0, 0, 1.22)
C3_HPR = Vec3(0, 0,0)


metadrive_simulation_state = namedtuple("metadrive_simulation_state", ["running", "done", "done_info"])
metadrive_vehicle_state = namedtuple("metadrive_vehicle_state", ["velocity", "position", "bearing", "steering_angle"])

def apply_metadrive_patches(arrive_dest_done=True):
  # By default, metadrive won't try to use cuda images unless it's used as a sensor for vehicles, so patch that in
  def add_image_sensor_patched(self, name: str, cls, args):
    if self.global_config["image_on_cuda"]:# and name == self.global_config["vehicle_config"]["image_source"]:
        sensor = cls(*args, self, cuda=True)
    else:
        sensor = cls(*args, self, cuda=False)
    assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
    self.sensors[name] = sensor

  EngineCore.add_image_sensor = add_image_sensor_patched

  # we aren't going to use the built-in observation stack, so disable it to save time
  def observe_patched(self, *args, **kwargs):
    return self.state

  ImageObservation.observe = observe_patched

  # disable destination, we want to loop forever
  def arrive_destination_patch(self, *args, **kwargs):
    return False

  if not arrive_dest_done:
    MetaDriveEnv._is_arrive_destination = arrive_destination_patch

def metadrive_process(dual_camera: bool, config: dict, camera_array, wide_camera_array, image_lock,
                      controls_recv: Connection, simulation_state_send: Connection, vehicle_state_send: Connection,
                      exit_event, op_engaged, test_duration, test_run):
  arrive_dest_done = config.pop("arrive_dest_done", True)
  apply_metadrive_patches(arrive_dest_done)

  road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
  if dual_camera:
    assert wide_camera_array is not None
    wide_road_image = np.frombuffer(wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

  env = MovingExampleEnv(config)
  #env = MetaDriveEnv(config)

  def get_current_lane_info(vehicle):
    _, lane_info, on_lane = vehicle.navigation._get_current_lane(vehicle)
    lane_idx = lane_info[2] if lane_info is not None else None
    return lane_idx, on_lane

  def reset():
    env.reset()
    env.vehicle.config["max_speed_km_h"] = 1000
    lane_idx_prev, _ = get_current_lane_info(env.vehicle)

    simulation_state = metadrive_simulation_state(
      running=True,
      done=False,
      done_info=None,
    )
    simulation_state_send.send(simulation_state)

    return lane_idx_prev

  lane_idx_prev = reset()
  start_time = None

  def get_cam_as_rgb(cam):
    cam = env.engine.sensors[cam]
    cam.get_cam().reparentTo(env.vehicle.origin)
    cam.get_cam().setPos(C3_POSITION)
    cam.get_cam().setHpr(C3_HPR)
    img = cam.perceive(to_float=False)
    if type(img) != np.ndarray:
      img = img.get() # convert cupy array to numpy
    return img



  steer_ratio = 8
  vc = [0,0]

  while not exit_event.is_set():
    vehicle_state = metadrive_vehicle_state(
      velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
      position=env.vehicle.position,
      bearing=float(math.degrees(env.vehicle.heading_theta)),
      steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING
    )
    vehicle_state_send.send(vehicle_state)

    if controls_recv.poll(0):
      while controls_recv.poll(0):
        steer_angle, gas, should_reset = controls_recv.recv()

      steer_metadrive = steer_angle * 1 / (env.vehicle.MAX_STEERING * steer_ratio)
      steer_metadrive = np.clip(steer_metadrive, -1, 1)

      vc = [steer_metadrive, gas]

      if should_reset:
        lane_idx_prev = reset()
        start_time = None

    is_engaged = op_engaged.is_set()
    global openpilot_engaged
    if is_engaged and start_time is None:
      openpilot_engaged = True
      start_time = time.monotonic()

    # ***************************************************************
    global aeb_alert, aeb_brake, fcw_alert, driver_alerted_time, driver_brake_action, driver_lateral_action, Panda_SafetyCheck_alert, hazard, hazard_time, hazard_type, OP_alertTime, OP_alertType_list, OP_alertText1_list, OP_alertText2_list, v1_rel_dis_t, v2_rel_dis_t, lead_rd_gt, out_of_lane_alert, driver_reaction_time, simulation_done, driver_lateral_alerted_time, recovery_mode_gas, recovery_mode_steer, ml_gas_action, ml_steer_action
    sm.update(0)

    alertText1 = sm['controlsState'].alertText1
    alertText2 = sm['controlsState'].alertText2
    alertType  = sm['controlsState'].alertType

    # LeadOne data from sensor
    dRel = sm['radarState'].leadOne.dRel
    yRel = sm['radarState'].leadOne.yRel #y means lateral direction
    vRel = sm['radarState'].leadOne.vRel
    vLead = sm['radarState'].leadOne.vLead
    if not sm['radarState'].leadOne.status:
      Lead_vehicle_in_vision = False
    else:
      Lead_vehicle_in_vision = True


    # Lead vehicle ground truth
    if vehicle1:
      v1_vel = math.sqrt(vehicle1.velocity[0]**2 + vehicle1.velocity[1]**2)
      v1_rel_dis_t = math.sqrt((vehicle1.position[0] - env.vehicle.position[0])**2 + (vehicle1.position[1] - env.vehicle.position[1])**2) - 5 # 5 meters vehicle length

    else:
      v1_vel = 0
      v1_rel_dis_t = 1000


    if vehicle2:
      v2_vel = math.sqrt(vehicle2.velocity[0]**2 + vehicle2.velocity[1]**2)
      v2_rel_dis_t = math.sqrt((vehicle2.position[0] - env.vehicle.position[0])**2 + (vehicle2.position[1] - env.vehicle.position[1])**2)
    else:
      v2_vel = 0
      v2_rel_dis_t = 1000

    #print(dRel, v1_rel_dis_t,v2_rel_dis_t)

    ego_vel = 0 # from sensor
    ego_vel_t = 0 # truth value from simulator

    if is_engaged:
      global frameIdx
      frameIdx += 1
      ego_vel =  math.sqrt(vehicle_state.velocity.x**2 + vehicle_state.velocity.y**2 + vehicle_state.velocity.z**2)
      ego_vel_t = math.sqrt(env.vehicle.velocity[0]**2 + env.vehicle.velocity[1]**2)

    # Lane position
    ylaneLines = []
    yroadEdges = []
    pathleft = pathright = 1
    laneLineleft=-1.85
    laneLineright = 1.85
    md = sm["modelV2"]
    if len(md.position.y)>0:
      yPos = round(md.position.y[0],2) # position
      ylaneLines = [round(md.laneLines[0].y[0],2),round(md.laneLines[1].y[0],2),round(md.laneLines[2].y[0],2),round(md.laneLines[3].y[0],2)]
      yroadEdges = [round(md.roadEdges[0].y[0],2), round(md.roadEdges[1].y[0],2)] #left and right roadedges
    if len(ylaneLines)>2:
      laneLineleft = ylaneLines[1]
      laneLineright = ylaneLines[2]
      pathleft = abs(laneLineleft) -0.65
      pathright = laneLineright
      #pathleft = yPos- laneLineleft
      #pathright = laneLineright-yPos
      roadEdgeLeft = yroadEdges[0]
      roadEdgeRight = yroadEdges[1]
      #print(pathleft)



    # ******************** Fault Injection Test ********************
    #if FI_test and rk.frame > 500 and (FI_flag == -1 or FI_flag > 0):
      #if dRel < 30 and dRel != 0 and is_engaged:
        #print('*******************FI****************')
        #FI_flag = 1
        #vc[1] = 0.6

    # ******************************************************
    if scenario == 5:
      cut = 1
    else:
      cut = 0
    if scenario == 6:
      move = 1
    else:
      move = 0

    if cut or move:
      if cut:
        if lane_change_start_frame != 0:
          lead_rd_gt = v2_rel_dis_t
        else:
          lead_rd_gt = v1_rel_dis_t
      if move:
        if lane_change_start_frame != 0:
          lead_rd_gt = v1_rel_dis_t
        else:
          lead_rd_gt = v2_rel_dis_t
    else:
      lead_rd_gt = v1_rel_dis_t
    #print(v1_rel_dis_t,v2_rel_dis_t,lead_rd_gt)

    # ************************** Driver Rection ********************************
    if Driver_react_Enable and driver_alerted_time == -1 and (fcw_alert or vc[1]<-0.95 or lead_rd_gt < 5 or (scenario == 5 and lane_change_start_frame != 0)): # or abs(patch.mean()>=0.15) #max gas//max brake//exceed speed limit/unsafe following distance<5m
      #(fcw_alert or ego_vel>1.1*vEgo*0.4407 or vc[1]>= 0.6 or vc[1]<-0.95 or v1_rel_dis_t<5)
    #if rk.frame > 2000 and driver_alerted_time == -1 and (fcw_alert or vc[1]>= 0.6 or vc[1]<-0.95 or dRel<5):
      driver_alerted_time = frameIdx
      print("driver alerted")

    laneLeft = env.vehicle.dist_to_left_side - 0.926
    laneRight = env.vehicle.dist_to_right_side - 0.926
    #print(laneLeft, pathleft)

    # Driver Reaction Brake
    if Driver_react_Enable == True:

      if driver_alerted_time >= 0 and frameIdx >=driver_reaction_time + driver_alerted_time: #average reaction time 2.5s
        driver_brake_action = True
        vc[1] = -driver_brake_simulator(frameIdx-driver_alerted_time-driver_reaction_time)#1
        print("Driver braked")
        #print("Driver:",frameIdx,driver_alerted_time,frameIdx-driver_alerted_time)
      else:
        driver_brake_action = False

    #Driver Reaction Lateral Control
    global out_of_lane_count, out_of_lane_count_max
    if Driver_lateral_react_Enable == True:
      if pathleft < 0.3 and openpilot_engaged:
          out_of_lane_count += 1
          if out_of_lane_count > out_of_lane_count_max:
            out_of_lane_count_max = out_of_lane_count
          print(out_of_lane_count)
          if out_of_lane_count > 5:
            out_of_lane_alert = True
      else:
          out_of_lane_count = 0
      if out_of_lane_alert and driver_lateral_alerted_time == -1:
        driver_lateral_alerted_time = frameIdx
      if driver_lateral_alerted_time != -1 and frameIdx > driver_lateral_alerted_time + driver_reaction_time and pathleft < 1:
        vc[0] = -0.003
        print("Driver Control")
        driver_lateral_action = True
      elif driver_lateral_alerted_time != -1 and frameIdx > driver_lateral_alerted_time + driver_reaction_time and pathleft >= 1:
        simulation_done = True
      else:
        driver_lateral_action == False




    # ********************* Define value used for AEB ********************************
    if aeb_independent and ego_vel:
      if cut and lane_change_start_frame != 0:
        rel_speed = v2_vel - ego_vel_t
      else:
        rel_speed = v1_vel - ego_vel_t
      ego_speed = ego_vel_t
      rel_distance = math.sqrt(sm['modelV2'].leadsV3[0].x[0]**2 + sm['modelV2'].leadsV3[0].y[0])
    else:
      rel_speed = vRel
      rel_distance = dRel
      ego_speed = ego_vel
    #***********************************************************
    if -rel_speed > 0:
      ttc = rel_distance / -rel_speed
    else:
      ttc = 1000

    treact = 2.5
    tfcw = treact + ego_speed / 4.5

    # 1st partial brake phase: decelerate at 4 m/s^2
    tpb1 = ego_speed / 3.8
    # 2nd partial brake phase: decelerate at 6 m/s^2
    tpb2 = ego_speed / 5.8
    # full brake phase: decelerate at 10 m/s^2
    tfb = ego_speed / 9.8

    #if ttc > tpb1 and ttc < tfcw:
    if ttc < tfcw:
      obstacle_count += 1
      if obstacle_count > 5:
        fcw_alert = True
    else:
      obstacle_count = 0
      fcw_alert = False


    # *********************************** AEB *******************************
    if AEB_React_Enable == True:
      if ttc > 0 and ttc < tfb:
        proc_brake = True
        aeb_alert = True
        aeb_brake = 1
      elif ttc > tfb and ttc < tpb2:
        proc_brake = True
        aeb_alert = True
        aeb_brake =0.95
      elif ttc > tpb2 and ttc < tpb1:
        proc_brake = True
        aeb_alert = True
        aeb_brake = 0.9
        #print(tpb2, ttc, tpb1, tfcw)


      if aeb_alert:
          vc[1] = -aeb_brake

    # ************************ Openpilot Alert *******************
    alertText1 = sm['controlsState'].alertText1
    alertText2 = sm['controlsState'].alertText2
    alertType  = sm['controlsState'].alertType

    if alertType and alertText1 and frameIdx != 0:
      OP_alertTime.append(frameIdx)
      OP_alertType_list.append(alertType)
      OP_alertText1_list.append(alertText1)

    # ************************ Satety Constraint Checker ****************
    #print(abs((vc[0] - env.vehicle.steering) * (env.vehicle.MAX_STEERING * steer_ratio))>0.5)
    #print(env.vehicle.MAX_STEERING * steer_ratio)
    if Panda_SafetyCheck_Enable and frameIdx > 1000 and aeb_alert != True and driver_brake_action != True and not alertType:

      acc = math.sqrt(md.acceleration.x[0]**2 + md.acceleration.y[0]**2 + md.acceleration.z[0]**2)
      if md.acceleration.x[0] < 0:
        acc *= -1
      #print(md.acceleration.x[0])
      if acc < -3.5 or acc > 2:
        Panda_SafetyCheck_alert = True
        vc[1] = 0

      '''
      if vc[1]<-0.875 or vc[1]>0.5:
          Panda_SafetyCheck_alert = True
          #vc[0] = 0
          vc[1] = 0
      '''

    # ********************** ML Model ********************
    global buffer, old_steer, old_gas
    if ml_model_Enable:
      if rk.frame == 1:
        model = load_model('lstm_model.h5', compile=False)
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
      if rk.frame % 5 == 0 and frameIdx > 1500:

        input_features = ['Speed', 'Rel Speed', 'Rel Distance', 'Path Left OP', 'Gas', 'Steer']
        scaler_x = MinMaxScaler(feature_range=(0, 1))


        input_array = [
          ego_speed,
          rel_speed,
          rel_distance,
          pathleft,
          old_gas,
          old_steer
        ]
        buffer.append(input_array)
        print(len(buffer))
        if len(buffer) > 20:
          buffer.pop(0)

        if len(buffer) == 20:
          input_buffer_array = np.array(buffer)

          # Scale input data to 0-1 range
          input_data_scaled = scaler_x.fit_transform(input_buffer_array)
          input_data_scaled = input_data_scaled.reshape((1, 20, len(input_features)))  # Reshape for LSTM input

          # Predict the Output using LSTM Model
          prediction = model.predict(input_data_scaled)

          # Construct Output Command for Metadrive
          gas_ml_output = float(prediction[0][0])
          steer_ml_output = float(prediction[0][1])

          delta_steer = abs(steer_ml_output - vc[0])
          delta_gas = abs(gas_ml_output - vc[1])
          print(delta_gas, delta_steer)
          global st_gas, st_steer
          st1_steer = max(0, st_steer + delta_steer - 0.019)
          st1_gas = max(0, st_gas + delta_gas - 0.4684)
          #print(st1_gas, st1_steer)

          if st1_steer > 0.03:
            recovery_mode_steer = True
            st1_steer = 0
          elif delta_steer < 0.019:
            recovery_mode_steer = False

          if st1_gas > 0.5:
            recovery_mode_gas = True
            st1_gas = 0
          elif delta_gas < 0.05:
            recovery_mode_gas = False

          if recovery_mode_steer:
            vc[0] = steer_ml_output
            ml_steer_action = True
          else:
            ml_steer_action = False

          if recovery_mode_gas:
            vc[1] = gas_ml_output
            ml_gas_action = True
          else:
            ml_gas_action = False

          old_gas = vc[1]
          old_steer = vc[0]
          st_steer = st1_steer
          st_gas = st1_gas



    global AEB_trigger_time, Driver_brake_trigger_time, Driver_lateral_trigger_time, FCW_alert_time, out_of_lane_alert_time
    if AEB_trigger_time == -1 and aeb_alert:
      AEB_trigger_time = frameIdx
    if Driver_brake_trigger_time == -1 and driver_brake_action:
      Driver_brake_trigger_time = frameIdx
    if Driver_lateral_trigger_time == -1 and driver_lateral_action:
      Driver_lateral_trigger_time = frameIdx
    if FCW_alert_time == -1 and fcw_alert:
      FCW_alert_time = frameIdx
    if out_of_lane_alert_time == -1 and out_of_lane_alert:
      out_of_lane_alert_time = frameIdx



    # *************** Label Hazard *****************
    hazrdType = 0
    if lead_rd_gt < 0.5 and lead_rd_gt > 0:
      hazrdType = 1
      hazard = True
      hazard_time.append(frameIdx)
      hazard_type.append(hazrdType)
    elif (pathleft < 0.1 or pathright <0.1) and len(ylaneLines)>2:
      hazrdType = 2
      hazard = True
      hazard_time.append(frameIdx)
      hazard_type.append(hazrdType)


    simulation_data = []
    if rk.frame % 5 == 0:
      # **************************** log file *************************
      if frameIdx > 500:
        simulation_data.append([frameIdx, ego_speed, rel_speed, rel_distance, lead_rd_gt, v1_rel_dis_t, v2_rel_dis_t, fcw_alert, ttc, tfcw,tpb1, tpb2, tfb, vc[0], vc[1], aeb_alert, driver_brake_action, driver_alerted_time, pathleft, pathright, laneLeft, laneRight, out_of_lane_alert, driver_lateral_alerted_time, driver_lateral_action, Panda_SafetyCheck_alert, hazard, hazrdType, alertType, alertText1, lane_change_start_frame, AEB_trigger_time, Driver_brake_trigger_time, Driver_lateral_trigger_time, ml_gas_action, ml_steer_action])
        df = pd.DataFrame(simulation_data, columns=['FrameIdx', 'Speed', ' Rel Speed', ' Rel Distance', 'Relative Distance Ground Truth', 'v1 Rel Dis GT', 'v2 Rel Dis GT', ' FCW', ' TTC', 'Tfcw','tpb1', 'tpb2', 'tfb', ' Steer', ' Gas', ' AEB: ', ' Driver Brake Action', 'driver alerted time', ' Path Left OP', 'Path Right OP', 'Lane Left MD', 'Lane Right MD', 'Out Of Lane Alert', 'driver_lateral_alerted_time', 'driver lateral action', 'Panda Safety Check', 'Hazard', 'Hazard Type', 'OP Alert Type', 'OP Alert Text', 'Lane Change Start Frame', 'AEB_trigger_time', 'Driver_brake_trigger_time', 'Driver_lateral_trigger_time', 'ML Gas Action', 'ML Steer Action'])
        file_path = f'results/DataRecord_{scenario}_{initial_dis}_{driver_reaction_time}.csv'
        header = not pd.io.common.file_exists(file_path)
        df.to_csv(file_path, mode='a',header=header, index=False)

        print('FrameIdx: ', frameIdx, 'Speed: ', ego_speed, 'Lead Speed', vLead, ' Rel Speed: ', rel_speed, ' Rel Distance: ', rel_distance, ' RD Ground Truth: ', lead_rd_gt, ' FCW: ', fcw_alert, ' Tfcw: ', tfcw, ' TTC: ', ttc, ' Steer: ', vc[0], ' Gas: ', vc[1], ' AEB: ', aeb_alert, ' Driver Alerted Time: ', driver_alerted_time, ' Driver Brake action: ', driver_brake_action, ' Driver Lateral Action: ', driver_lateral_action, ' Panda Safety Check: ', Panda_SafetyCheck_alert, ' Path Left: ', pathleft, ' cut frame: ', lane_change_start_frame)


    if rk.frame % 5 == 0:
      _, _, terminated, _, _ = env.step(vc)
      timeout = True if start_time is not None and time.monotonic() - start_time >= test_duration else False
      lane_idx_curr, on_lane = get_current_lane_info(env.vehicle)
      out_of_lane = lane_idx_curr != lane_idx_prev or not on_lane
      lane_idx_prev = lane_idx_curr
      #if (aeb_alert or driver_brake_action) and ego_speed < 0.3:
        #vehicle_stop = True
      #else:
        #vehicle_stop = False

      if frameIdx > 1000 and ego_speed < 0.3:
        vehicle_stop = True
      else:
        vehicle_stop = False

      if terminated or ((out_of_lane or timeout) and test_run) or vehicle_stop or simulation_done:
        if terminated:
          done_result = env.done_function("default_agent")
        elif out_of_lane:
          done_result = (True, {"out_of_lane" : True})
        elif timeout:
          done_result = (True, {"timeout" : True})
        elif vehicle_stop:
          done_result = (True, {"vehicle stop" : True})
        elif simulation_done:
          done_result = (True, {"done" : True})

        simulation_state = metadrive_simulation_state(
          running=False,
          done=done_result[0],
          done_info=done_result[1],
        )
        # ****************** Summary file **************************
        if frameIdx > 500:
          dfSummary = []
          dfSummary.append([frameIdx, done_result[0], done_result[1],FCW_alert_time, AEB_trigger_time, Driver_brake_trigger_time, out_of_lane_alert_time, Driver_lateral_trigger_time, out_of_lane_count_max, hazard, hazard_time, hazard_type, OP_alertTime, OP_alertType_list, OP_alertText1_list])
          dfSummary = pd.DataFrame(dfSummary, columns=['FrameIdx', 'Done', 'Result', 'FCW_alert_time', 'AEB_trigger_time', 'Driver_brake_trigger_time', 'out_of_lane_alert_time', 'Driver_lateral_trigger_time', 'out_of_lane_count_max', 'Hazard', 'Hazard Time', 'Hazard Type', 'OP Alert Time', 'OP Alert Type', 'OP Alert Text'])
          dfSummary.to_csv(f'results/Summary_{scenario}_{initial_dis}_{driver_reaction_time}.csv', mode='a',header=not pd.io.common.file_exists(f'results/Summary_{scenario}_{initial_dis}_{driver_reaction_time}.csv'), index=False)
          print(done_result,hazard, hazard_time, hazard_type,pathleft,pathright)

        simulation_state_send.send(simulation_state)


      if dual_camera:
        wide_road_image[...] = get_cam_as_rgb("rgb_wide")
      road_image[...] = get_cam_as_rgb("rgb_road")
      image_lock.release()

    rk.keep_time()
