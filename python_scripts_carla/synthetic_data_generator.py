#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

from __future__ import print_function
import queue
import sys
import time
import cv2
import carla

from carla import ColorConverter as cc

import argparse
import logging
import random
import re
import math

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    presets = ['Default']
    presets = [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
    return presets

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        bps_list = [x for x in bps if x.id not in [
            'vehicle.carlamotors.carlacola',
            'vehicle.tesla.cybertruck',
            'vehicle.micro.microlino'
        ]]
        return bps_list

    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

class WorldManager(object):

    def __init__(self, carla_client, carla_world, args):
        self.client = carla_client
        self.world = carla_world
        self.world_init_settings = carla_world.get_settings()
        self.tm = carla_client.get_trafficmanager()
        self.sync = args.sync
        self._ego = None
        self.sensor_manager = None
        self.vehicle_traffic_manager = None
        self.pedestrian_traffic_manager = None
        # Training data maps
        # self.maps = ["Town01"]
        # self.maps = ["Town02"]
        # self.maps = ["Town03"] # more vehicles?
        # self.maps = ["Town04"] # might need to be unused OR more vehicles + more pedestrians (too few vehicles)
        self.maps = ["Town10HD"]
        # self.maps = ["Town05"]*
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._cam_width = args.width
        self._cam_height = args.height
        self._ego_blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))

    def set_weather(self, index):
        mod_index = index % len(self._weather_presets)
        preset = self._weather_presets[mod_index]
        self._ego.get_world().set_weather(preset[0])

    def set_map(self, index):
        self.client.load_world(self.maps[index], reset_settings=True)
        self.reset_settings()
        time.sleep(8)
        print('Finished sleeping after loading map.')
        self.apply_settings()
        self.world = self.client.get_world()
        self.tm = self.client.get_trafficmanager()
        self.map = self.world.get_map()
        try:
            self.map_available_waypoints = self.map.generate_waypoints(10.0)
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def _spawn_ego(self):
        if self._ego_blueprint.has_attribute('role_name'):
            self._ego_blueprint.set_attribute('role_name', 'hero')
        if self._ego_blueprint.has_attribute('color'):
            color = random.choice(self._ego_blueprint.get_attribute('color').recommended_values)
            self._ego_blueprint.set_attribute('color', color)
        if self._ego_blueprint.has_attribute('driver_id'):
            driver_id = random.choice(self._ego_blueprint.get_attribute('driver_id').recommended_values)
            self._ego_blueprint.set_attribute('driver_id', driver_id)
        if self._ego_blueprint.has_attribute('is_invincible'):
            self._ego_blueprint.set_attribute('is_invincible', 'true')

        spawn_point = carla.Transform()
        if self.map_available_waypoints:
            spawn_point = random.choice(self.map_available_waypoints)        
            self.map_available_waypoints.remove(spawn_point)
            spawn_point = spawn_point.transform

        while self._ego is None:
            spawn_point.location.z += 0.1
            self._ego = self.world.try_spawn_actor(self._ego_blueprint, spawn_point)
            self.modify_vehicle_physics(self._ego)
        spectator_transform = spawn_point
        spectator_transform.location.z += 2
        # spectator_transform.rotation.pitch -= 90.0
        self.world.get_spectator().set_transform(spectator_transform)
        self.world.tick()

    def _spawn_sensors(self):
        if self.sensor_manager is None: 
            if self._ego is not None:
                self.sensor_manager = SensorsManager(self.world, self._cam_width, self._cam_height)
            else:
                print('Ego vehicle spawn --> Sensors spawn.')
                exit(-1)
        self.sensor_manager.transform_index = 0
        self.sensor_manager.spawn_sensors(self._ego)

    def _spawn_vehicles(self):
        if self.vehicle_traffic_manager is None:
            if self._ego is not None:
            # TODO: Parametrize vehicle count
                self.vehicle_traffic_manager = VehicleTrafficManager(self.world, vehicle_count=40)
            else:
                print('Ego vehicle spawn --> Sensors spawn.')
                exit(-1)

        self.vehicle_traffic_manager.world = self.world
        self.vehicle_traffic_manager.spawn_vehicles(spawn_points=self.map_available_waypoints)

    def _spawn_pedestrians(self):
        if self.pedestrian_traffic_manager is None:
            self.pedestrian_traffic_manager = PedestrianTrafficManager(self.client, self.world)
        ego_transform = self._ego.get_transform()
        ego_transform.location.z += 2.0
        self.world.tick()

        self.pedestrian_traffic_manager.world = self.world
        self.pedestrian_traffic_manager.client = self.client
        self.pedestrian_traffic_manager.spawn_pedestrians(pedestrian_number=random.randrange(15, 20, 1),
                                                          pedestrian_crossing_perc=random.random() * 0.001,
                                                          pedestrian_running_perc=random.random() * 0.005,
                                                          ego_sensor_transform=ego_transform,
                                                          ego_sensor_fov=60)

    def _despawn_ego(self):
        if self._ego is not None:
            print(self._ego.destroy())
            self._ego = None
        self.world.tick()

    def _despawn_sensors(self):
        self.sensor_manager.despawn_sensors()

    def _despawn_vehicles(self):
        self.vehicle_traffic_manager.despawn_vehicles()

    def _despawn_pedestrians(self):
        self.pedestrian_traffic_manager.despawn_pedestrians()

    def apply_settings(self, map_index=None):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        if map_index is not None and self.maps[map_index].equals("Town12"):
            settings.tile_stream_distance = 100
            settings.actor_active_distance = 100
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(True)

    def reset_settings(self):
        self.world.apply_settings(self.world_init_settings)
        self.tm.set_synchronous_mode(False)
            
    def spawn_actors(self):
        self._spawn_ego()
        self._spawn_pedestrians()
        self._spawn_vehicles()
        self._spawn_sensors()

    def despawn_actors(self):
        self._despawn_sensors()
        self._despawn_vehicles()
        self._despawn_pedestrians()
        self._despawn_ego()

        print('Pedestrians left:\t', len(self.world.get_actors().filter("walker.*")))
        print('Vehicles left:\t\t', len(self.world.get_actors().filter("vehicle.*")))

class VehicleTrafficManager(object):
    def __init__(self, world, vehicle_count) -> None:
        self.world = world
        self.vehicle_count = vehicle_count
        self.spawned_vehicles = []

    def spawn_vehicles(self, spawn_points):
        shuffled_spawn_points = spawn_points.copy()
        random.shuffle(shuffled_spawn_points)
        
        if self.vehicle_count > len(shuffled_spawn_points):
            print('More vehicles than available spawn points!')
            exit(-1)

        vehicle_blueprints = get_actor_blueprints(self.world, "vehicle.*", "All")
        for _ in range(self.vehicle_count):   
            spawn_point = random.choice(shuffled_spawn_points)
            shuffled_spawn_points.remove(spawn_point)
            spawn_point = spawn_point.transform
            vehicle_blueprint = random.choice(vehicle_blueprints)

            if vehicle_blueprint.has_attribute('color'):
                color = random.choice(vehicle_blueprint.get_attribute('color').recommended_values)
                vehicle_blueprint.set_attribute('color', color)
            if vehicle_blueprint.has_attribute('driver_id'):
                driver_id = random.choice(vehicle_blueprint.get_attribute('driver_id').recommended_values)
                vehicle_blueprint.set_attribute('driver_id', driver_id)
            if vehicle_blueprint.has_attribute('is_invincible'):
                vehicle_blueprint.set_attribute('is_invincible', 'true')

            vehicle = None
            trial = 0
            while vehicle is None and trial < 40:
                spawn_point.location.z += 0.01
                vehicle = self.world.try_spawn_actor(vehicle_blueprint, spawn_point)
                trial += 1
            self.spawned_vehicles += [vehicle]
    
        # Settle spawned vehicles
        for _ in range(10):
            self.world.tick()

    def despawn_vehicles(self):
        for spawned_vehicle in self.spawned_vehicles:
            if spawned_vehicle is not None:
                try:
                    spawned_vehicle.destroy()
                except:
                    pass
                finally:
                    self.world.tick()
                spawned_vehicle = None
        self.spawned_vehicles = []
        self.world.tick()

class PedestrianTrafficManager(object):
    
    SpawnActor = carla.command.SpawnActor
    DestroyActor = carla.command.DestroyActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    def __init__(self, client, world) -> None:
        self.client = client
        self.world = world
        self.pedestrian_actors = []
        self.pedestrian_actors_ids = []
        self.pedestrian_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        self.pedestrian_bp_index = 0
        self.pedestrian_bps = get_actor_blueprints(world, 'walker.pedestrian.*', 'All')
        random.shuffle(self.pedestrian_bps)
        
    def _setup_pedestrian_spawn_data(self, spawn_point, pedestrian_running_perc):
        pedestrian_spawn_data = {
            'bp': self.pedestrian_bps[self.pedestrian_bp_index],
            'speed': None,
            'spawn_point': spawn_point
        }
        if (pedestrian_spawn_data['bp'].has_attribute('is_invincible')):
            pedestrian_spawn_data['bp'].set_attribute('is_invincible', 'false')
        if (pedestrian_spawn_data['bp'].has_attribute('speed')):
            if (random.random() > pedestrian_running_perc):
                pedestrian_spawn_data['speed'] = pedestrian_spawn_data['bp'].get_attribute('speed').recommended_values[1]
            else:
                pedestrian_spawn_data['speed'] = pedestrian_spawn_data['bp'].get_attribute('speed').recommended_values[2]
        else:
            pedestrian_spawn_data['speed'] = 0.0
        self.pedestrian_bp_index = (self.pedestrian_bp_index + 1) % len(self.pedestrian_bps)
        return pedestrian_spawn_data
    
    def _launch_pedestrian_spawn_command_batch(self, pedestrian_spawn_data_list):
        pedestrian_spawn_command_batch = []
        for pedestrian_spawn_data in pedestrian_spawn_data_list:
            pedestrian_spawn_command_batch.append(PedestrianTrafficManager.SpawnActor(
                pedestrian_spawn_data['bp'],
                pedestrian_spawn_data['spawn_point']
            ))
        return self.client.apply_batch_sync(pedestrian_spawn_command_batch, True)

    def _check_pedestrian_spawn_results(self, pedestrian_spawn_data_list, pedestrian_spawn_results):
        pedestrian_spawned_check_data_list = []
        for i in range(len(pedestrian_spawn_results)):
            if pedestrian_spawn_results[i].error:
                pass
            else:
                pedestrian_spawned_check_data_list.append({
                    'id': pedestrian_spawn_results[i].actor_id,
                    'speed': pedestrian_spawn_data_list[i]['speed']
                })
        return pedestrian_spawned_check_data_list
    
    def _launch_pedestrian_controller_spawn_command_batch(self, pedestrian_spawn_data_list):
        pedestrian_controller_spawn_command_batch = []
        for i in range(len(pedestrian_spawn_data_list)):
            pedestrian_controller_spawn_command_batch.append(
                PedestrianTrafficManager.SpawnActor(
                    self.pedestrian_controller_bp, 
                    carla.Transform(),
                    pedestrian_spawn_data_list[i]['id']
                )
            )
        return self.client.apply_batch_sync(pedestrian_controller_spawn_command_batch, True)
    
    def _check_pedestrian_controller_spawn_results(self, pedestrian_spawn_data_list, pedestrian_controller_spawn_results):
        pedestrian_controller_spawned_check_data_list = []
        for i in range(len(pedestrian_controller_spawn_results)):
            if pedestrian_controller_spawn_results[i].error:
                pass
            else:
                pedestrian_controller_spawned_check_data_list.append({
                    'con':pedestrian_controller_spawn_results[i].actor_id,
                    'id': pedestrian_spawn_data_list[i]['id'],
                    'speed': pedestrian_spawn_data_list[i]['speed']
                })
        return pedestrian_controller_spawned_check_data_list
        
    def _store_pedestrian_actors(self, pedestrian_spawn_data_list):
        pedestrian_related_ids = []
        for i in range(len(pedestrian_spawn_data_list)):
            pedestrian_related_ids.append(pedestrian_spawn_data_list[i]['con'])
            pedestrian_related_ids.append(pedestrian_spawn_data_list[i]['id'])
        self.pedestrian_actors = self.world.get_actors(pedestrian_related_ids)
        self.pedestrian_actors_ids = pedestrian_related_ids

    def _apply_pedestrian_control(self, pedestrian_spawn_data_list):
        for i in range(0, len(self.pedestrian_actors_ids), 2):
            self.pedestrian_actors[i].start()
            self.pedestrian_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            self.pedestrian_actors[i].set_max_speed(float(pedestrian_spawn_data_list[int(i/2)]['speed']))
        self.world.tick()

    def spawn_pedestrians(self, pedestrian_number, pedestrian_crossing_perc, pedestrian_running_perc, ego_sensor_transform, ego_sensor_fov):
        spawn_points = []

        cam_front = ego_sensor_transform.get_forward_vector()
        cam_front_2d = carla.Vector2D(cam_front.x, cam_front.y).make_unit_vector()
        cam_right = ego_sensor_transform.get_right_vector()

        for _ in range(pedestrian_number):
        
            spawn_point = carla.Transform()

            pedestrian_good_candidate = False
            trial = 0
            while pedestrian_good_candidate is not True and trial < 100:
                spawn_point_location = self.world.get_random_location_from_navigation()
                spawn_point_location.z += 0.1 # offset on up axis for avoiding invalid object intersections

                cam_pedestrian_ray = spawn_point_location - ego_sensor_transform.location
                cam_pedestrian_ray_2d = carla.Vector2D(cam_pedestrian_ray.x, cam_pedestrian_ray.y).make_unit_vector()
                intersected_objects = self.world.cast_ray(ego_sensor_transform.location, spawn_point_location)
                d = cam_pedestrian_ray_2d.x * cam_front_2d.x + cam_pedestrian_ray_2d.y * cam_front_2d.y
                
                pedestrian_good_candidate = cam_pedestrian_ray.length() > 10.0 and cam_pedestrian_ray.length() < 120.0 and len(intersected_objects) == 0 and d > 0 #and (cam_front_pedestrian_ray_angle < ego_sensor_fov/2 and cam_front_pedestrian_ray_angle > -ego_sensor_fov/2)
                
                trial += 1
                # if pedestrian_good_candidate:
                # self.world.debug.draw_arrow(ego_sensor_transform.location, spawn_point_location, color=carla.Color(0,255,200), arrow_size=0.5)
                # self.world.tick()

            if (spawn_point_location != None):
                spawn_point.location = spawn_point_location
                spawn_points.append(spawn_point)

        pedestrian_spawn_data_list = []
        for spawn_point in spawn_points:
            pedestrian_spawn_data_list.append(self._setup_pedestrian_spawn_data(spawn_point=spawn_point, pedestrian_running_perc=pedestrian_running_perc))
        
        self.world.set_pedestrians_cross_factor(pedestrian_crossing_perc)
        pedestrian_spawn_results = self._launch_pedestrian_spawn_command_batch(pedestrian_spawn_data_list)
        pedestrian_spawned_check_data_list = self._check_pedestrian_spawn_results(pedestrian_spawn_data_list, pedestrian_spawn_results)

        pedestrian_controller_spawn_results = self._launch_pedestrian_controller_spawn_command_batch(pedestrian_spawned_check_data_list)
        pedestrian_controller_spawned_check_data_list = self._check_pedestrian_controller_spawn_results(pedestrian_spawned_check_data_list, pedestrian_controller_spawn_results)
        
        self._store_pedestrian_actors(pedestrian_controller_spawned_check_data_list)
        self._apply_pedestrian_control(pedestrian_controller_spawned_check_data_list)

        # Tick for pedestrians to settle
        print(f'Pedestrians spawned: {len(self.pedestrian_actors_ids)/2}')
        for _ in range(10):
            self.world.tick()   

    def despawn_pedestrians(self):
        pedestrian_destroy_command_batch = []

        for i in range(0, len(self.pedestrian_actors_ids), 2):
            self.pedestrian_actors[i].stop()

        for pedestrian_actor_id in self.pedestrian_actors_ids:
            pedestrian_destroy_command_batch.append(PedestrianTrafficManager.DestroyActor(pedestrian_actor_id))
        self.client.apply_batch_sync(pedestrian_destroy_command_batch, True)

        self.pedestrian_actors = []
        self.pedestrian_actors_ids = []

class SensorsManager(object):
    SEM_CURRENT_FRAME = 2500
    RGB_CURRENT_FRAME = 2500

    def __init__(self, world, width, height):
        self._parent = None
        self.world = world
        self.queue_dict = {}
        self.width = width
        self.height = height
        self.bp_library = None
        self.sensor_types = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
        ]
        
        self.index = None

    def despawn_sensors(self):
        if self.sensor_rgb is not None:
            self.sensor_rgb.stop()
            self.sensor_rgb.destroy()
            self.sensor_rgb = None
        if self.sensor_semseg is not None:
            self.sensor_semseg.stop()    
            self.sensor_semseg.destroy()
            self.sensor_semseg = None

    def spawn_sensors(self, parent_actor):
        self._parent = parent_actor

        if self.bp_library is None:
            self.bp_library = self._parent.get_world().get_blueprint_library()
            # TODO: Solve hardcoded values

            # Epsilon value is subtracted to ensure that there is no sensor capture miss because of world tick rate
            # RGB sensor blueprint fetch
            self.bp_sensor_rgb = self.bp_library.find('sensor.camera.rgb')
            self.bp_sensor_rgb.set_attribute('image_size_x', str(self.width))
            self.bp_sensor_rgb.set_attribute('image_size_y', str(self.height))
            self.bp_sensor_rgb.set_attribute('bloom_intensity', str(0.0))
            self.bp_sensor_rgb.set_attribute('fstop', str(1.8))
            self.bp_sensor_rgb.set_attribute('sensor_tick', str(0.0))
            self.bp_sensor_rgb.set_attribute('fov', '60')
            # Semantic segmentation blueprint fetch
            self.bp_sensor_semseg = self.bp_library.find('sensor.camera.semantic_segmentation')
            self.bp_sensor_semseg.set_attribute('image_size_x', str(self.width))
            self.bp_sensor_semseg.set_attribute('image_size_y', str(self.height))
            self.bp_sensor_semseg.set_attribute('sensor_tick', str(0.0))
            self.bp_sensor_semseg.set_attribute('fov', '60')
            
        vehicle_bound_x = 0.5 + self._parent.bounding_box.extent.x
        vehicle_bound_y = 0.5 + self._parent.bounding_box.extent.y
        vehicle_bound_z = 0.5 + self._parent.bounding_box.extent.z
        camera_transform = carla.Transform(carla.Location(x=+0.8*vehicle_bound_x, y=+0.0*vehicle_bound_y, z=1.3*vehicle_bound_z))

        camera_transform.rotation.pitch -= 2.5
        # camera_transform.rotation.yaw +=  random.uniform(-2.0, 2.0)
        # camera_transform.rotation.roll += random.uniform(-2.0, 2.0)

        self.sensor_rgb = self._parent.get_world().spawn_actor(
                self.bp_sensor_rgb,
                camera_transform,
                attach_to=self._parent,
                attachment_type=carla.AttachmentType.Rigid)
        self.sensor_semseg = self._parent.get_world().spawn_actor(
                self.bp_sensor_semseg,
                camera_transform,
                attach_to=self._parent,
                attachment_type=carla.AttachmentType.Rigid)

        self.world.tick()
        self.__listen_sensors()
         
    def __listen_sensors(self):
        self.queue_dict['rgb'] = queue.Queue()
        self.queue_dict['semantic_segmentation'] = queue.Queue()
        self.sensor_rgb.listen(self.queue_dict['rgb'].put)
        self.sensor_semseg.listen(self.queue_dict['semantic_segmentation'].put)
    
    @staticmethod
    def parse_image(data):
        # TODO: Parametrize hardcoded path strings
        if data["name"] == 'semantic_segmentation':
            array = np.frombuffer(data["image"].raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data["image"].height, data["image"].width, 4))
            array = array[:, :, 2]
            cv2.imwrite(f'C:\\Users\\Manuel\\Projects\\GitHub_Repositories\\master_thesis\\datasets\\synthetic\\train\\{data["name"]}\\synthetic_{data["name"]}_{SensorsManager.SEM_CURRENT_FRAME}.png', array)
            SensorsManager.SEM_CURRENT_FRAME += 1
        elif data["name"] == 'rgb':
            array = np.frombuffer(data["image"].raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data["image"].height, data["image"].width, 4))
            array = array[:, :, :3]
            cv2.imwrite(f'C:\\Users\\Manuel\\Projects\\GitHub_Repositories\\master_thesis\\datasets\\synthetic\\train\\{data["name"]}\\synthetic_{data["name"]}_{SensorsManager.RGB_CURRENT_FRAME}.png', array)
            SensorsManager.RGB_CURRENT_FRAME += 1

def simulation_loop(args):
    
    world = None
    number_of_images = 500

    client = carla.Client(args.host, args.port)
    client.set_timeout(200.0)
    sim_world = client.get_world()

    world = WorldManager(client, sim_world, args)

    try:
        map_index = -1
        for image_index in range(number_of_images):
            world.apply_settings()

            new_map_index = int(image_index / (number_of_images / len(world.maps)))
            if new_map_index != map_index:
                # TODO: Fix sleeping with a wait on the currently active actors (especially cameras)
                time.sleep(1.5)
                map_index = new_map_index
                world.set_map(map_index)
                print('------------------------- New map -------------------------')

            world.spawn_actors()
            world.set_weather(image_index)

            world.world.tick()
            time.sleep(1)

            print('Map index:\t', map_index)
            print('Image index:\t', image_index)
            print('Cameras:\t', len(world.sensor_manager.queue_dict.items()))
            
            for queue_key, queue in world.sensor_manager.queue_dict.items():
                print(f'\t{queue_key} received images', queue.qsize())
                while not queue.empty():
                    SensorsManager.parse_image({'image': queue.get(), 'name': queue_key})

            world.despawn_actors()
        world.reset_settings()

    except KeyboardInterrupt:
        world.reset_settings()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='2048x1024',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='ego',
        help='actor role name (default: "ego")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        default='True',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        simulation_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':

    main()
