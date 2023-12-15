import numpy as np

import gym_auv.utils.geomutils as geom
import gym_auv.utils.helpers as helpers
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.path import RandomCurveThroughOrigin, Path
from gym_auv.objects.obstacles import PolygonObstacle, VesselObstacle, CircularObstacle
from gym_auv.environment import BaseEnvironment
from gym_auv.objects.rewarder import ColregRewarder, ColavRewarder, ColavRewarder2, PathRewarder, SafetyColavRewarder
import shapely.geometry, shapely.errors

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

vessel_speed_vals = np.loadtxt('resources/speed_vals.txt')
vessel_speed_density = np.loadtxt('resources/speed_density.txt')

class MovingObstacles(BaseEnvironment):

    def __init__(self, *args, **kwargs) -> None:
        """
        Sets following parameters for the scenario before calling super init. method:
            self._n_moving_obst : Number of moving obstacles
            self._n_static_obst : Number of static obstacles
            self._rewarder_class : Rewarder used, e.g. PathRewarder, ColregRewarder
        """
        super().__init__(*args, **kwargs)

    def _generate(self):
        #self.rng = np.random.RandomState(3) # ADDED, USE ONLY WHEN PLOTTING MULTIPLETRAJS, ELSE COMMENT OUT

        # Initializing path
        if not hasattr(self, '_n_waypoints'):
            self._n_waypoints = int(np.floor(4*self.rng.rand() + 2))

        self.path = RandomCurveThroughOrigin(self.rng, self._n_waypoints, length=300)

        # Initializing vessel (vessel params used to set other obstacles also)
        init_state = self.path(0)
        init_angle = self.path.get_direction(0)
        init_state[0] += 50*(self.rng.rand()-0.5)
        init_state[1] += 50*(self.rng.rand()-0.5)
        init_angle = geom.princip(init_angle + 2*np.pi*(self.rng.rand()-0.5))
        init_state[0] += 50*(self.rng.rand()-0.5)
        init_state[1] += 50*(self.rng.rand()-0.5)
        init_angle = geom.princip(init_angle + 2*np.pi*(self.rng.rand()-0.5))
        self.vessel = Vessel(self.config, np.hstack([init_state, init_angle]), width=self.config["vessel_width"])
        prog = 0
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog
        
        self.obstacles = []

        # Adding moving obstacles
        for _ in range(self._n_moving_obst):
            other_vessel_trajectory = []

            obst_position, obst_radius = helpers.generate_obstacle(self.rng, self.path, self.vessel, obst_radius_mean=10, displacement_dist_std=500)
            obst_direction = self.rng.rand()*2*np.pi
            obst_speed = self.rng.choice(vessel_speed_vals, p=vessel_speed_density)
            #obst_speed = np.random.choice(vessel_speed_vals, p=vessel_speed_density)

            for i in range(10000):
                other_vessel_trajectory.append((i, (
                    obst_position[0] + i*obst_speed*np.cos(obst_direction), 
                    obst_position[1] + i*obst_speed*np.sin(obst_direction)
                )))
            other_vessel_obstacle = VesselObstacle(width=obst_radius, trajectory=other_vessel_trajectory)
            self.obstacles.append(other_vessel_obstacle)

        # Adding static obstacles
        if not hasattr(self,'displacement_dist_std'):
            self.displacement_dist_std = 250

        for _ in range(self._n_static_obst):
            obstacle = CircularObstacle(*helpers.generate_obstacle(self.rng, self.path, self.vessel, displacement_dist_std=self.displacement_dist_std))
            self.obstacles.append(obstacle)
        
        # Perturb initial vessel position and heading (ONLY USED WHEN PLOTTING MULTIPLE INIT POS IN PLOT MANY TRAJ TESTING, COMMENT OUT IF NOT)
        '''
        rng_vessel_init = np.random.RandomState(np.random.randint(1000000)) # ADDED
        init_state[0] += 250*(rng_vessel_init.rand()-0.5)
        init_state[1] += 250*(rng_vessel_init.rand()-0.5)
        init_angle = geom.princip(init_angle + 2*np.pi*(rng_vessel_init.rand()-0.5)) # Overwrite with random heading for given vessel
        self.vessel = Vessel(self.config, np.hstack([init_state, init_angle]), width=self.config["vessel_width"])
        '''
        self._update()

class MovingObstaclesNoRules(MovingObstacles):
    def __init__(self, *args, **kwargs):
        self._n_moving_obst = 17
        self._n_static_obst = 11
        self._rewarder_class = PathRewarder  # ColavRewarder2
        super().__init__(*args, **kwargs)

class MovingObstaclesColreg(MovingObstacles):
    def __init__(self, *args, **kwargs):
        self._n_moving_obst = 17
        self._n_static_obst = 11
        self._rewarder_class = ColregRewarder
        super().__init__(*args, **kwargs)


########################################### THOMAS' CUSTOM ENVS ########################################################
class Env0(MovingObstacles):
    '''
    Complexity index: 1 (trivial case)
        Path is a straight line in a random direction.
        No static obstacles.
        No moving obstacles.
    '''
    def __init__(self, *args, **kwargs):
        self.straight_path = True
        self._n_waypoints = -1  # less than 2 --> only start-/end-point --> straight path
        self._n_moving_obst = 0
        self._n_static_obst = 0
        self._rewarder_class = PathRewarder
        super().__init__(*args, **kwargs)


class Env1(MovingObstacles):
    '''
    Complexity index: 2
        Path is a simple random curve.
        No static obstacles.
        No moving obstacles.
    '''
    def __init__(self, *args, **kwargs):
        self._n_waypoints = 2  # Curve complexity: more waypoints --> more turns. Remove to get random complexity.
        self._n_moving_obst = 0
        self._n_static_obst = 0
        self._rewarder_class = PathRewarder
        super().__init__(*args, **kwargs)


class Env2(MovingObstacles):
    '''
    Complexity index: 3
        Path is a straight line in a random direction.
        Some static obstacles.
        No moving obstacles.
    '''
    def __init__(self, *args, **kwargs):
        self.straight_path = True
        self._n_waypoints = -1  # Curve complexity: more waypoints --> more turns. Remove to get random complexity.
        self._n_moving_obst = 0
        self._n_static_obst = 4
        self.displacement_dist_std = 0  # Object distance from path
        self._rewarder_class = PathRewarder  # ColavRewarder
        super().__init__(*args, **kwargs)


class Env3(MovingObstacles):
    '''
    Complexity index: 4
        Path is a straight line in a random direction.
        Some static obstacles.
        Some moving obstacles.
    '''
    def __init__(self, *args, **kwargs):
        self.straight_path = True
        self._n_waypoints = -1  # Curve complexity: more waypoints --> more turns. Remove to get random complexity.
        self._n_moving_obst = 17
        self._n_static_obst = 4
        self.displacement_dist_std = 0  # Object distance from path
        self._rewarder_class = PathRewarder
        super().__init__(*args, **kwargs)

class Env4(MovingObstacles):
    '''
    Complexity index: 5 (almost the same as MovingObstaclesNoRules)
        Path is a simple random curve.
        Some static obstacles.
        Some moving obstacles.
    '''
    def __init__(self, *args, **kwargs):
        self._n_waypoints = 2  # Curve complexity: more waypoints --> more turns. Remove to get random complexity.
        self._n_moving_obst = 17
        self._n_static_obst = 4
        self.displacement_dist_std = 0  # Object distance from path
        self._rewarder_class = PathRewarder
        super().__init__(*args, **kwargs)


########################################### SAFETY FILTER ENVS ########################################################
class RandomScenario1(MovingObstacles):
    '''
    Complexity index: 3
        Path is a straight line in a random direction.
        Some static obstacles.
        No moving obstacles.
    '''
    def __init__(self, *args, **kwargs):
        self.straight_path = True
        self._n_waypoints = 2 # Curve complexity: more waypoints --> more turns. Remove to get random complexity.
        self.n_moving_obst = 5
        self.n_static_obst = 5
        self.displacement_dist_std = 100 #150
        self.static_obst_radius_mean = 25 #30
        self.moving_obst_radius_mean = 15 #20
        self._rewarder_class = SafetyColavRewarder
        super().__init__(*args, **kwargs)
        
        
    def _generate(self):

        safety_filter_rank = -1
        if hasattr(self.vessel, 'safety_filter_rank'):
            safety_filter = self.vessel.safety_filter
            safety_filter_rank = self.vessel.safety_filter_rank

        # Initializing path

        self.path = RandomCurveThroughOrigin(self.rng, self._n_waypoints, length=500)

        # Initializing vessel
        init_state = self.path(0)
        init_angle = self.path.get_direction(0)

        #Random State
        #init_state[0] += 50*(self.rng.rand()-0.5)
        #init_state[1] += 50*(self.rng.rand()-0.5)
        #init_angle = geom.princip(init_angle + 2*np.pi*(self.rng.rand()-0.5))

        self.vessel = Vessel(self.config, np.hstack([init_state, init_angle]), width=self.config["vessel_width"])
        prog = 0
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        min_distance_to_path = 10
        
        self.obstacles = []

        # Adding moving obstacles
        for _ in range(self.n_moving_obst):
            other_vessel_trajectory = []

            obst_position, obst_radius = helpers.generate_obstacle(self.rng, self.path, self.vessel, obst_radius_mean=self.moving_obst_radius_mean, displacement_dist_std=self.displacement_dist_std)
            obst_direction = self.rng.rand()*2*np.pi
            obst_speed = np.random.choice(vessel_speed_vals, p=vessel_speed_density)

            for i in range(10000):
                other_vessel_trajectory.append((i, (
                    obst_position[0] + i*obst_speed*np.cos(obst_direction), 
                    obst_position[1] + i*obst_speed*np.sin(obst_direction)
                )))
            other_vessel_obstacle = VesselObstacle(width=obst_radius, trajectory=other_vessel_trajectory)
            self.obstacles.append(other_vessel_obstacle)

        # Adding static obstacles

        for o in range(self.n_static_obst):
            obstacle = CircularObstacle(*helpers.generate_obstacle(self.rng, self.path, self.vessel, displacement_dist_std=self.displacement_dist_std, obst_radius_mean=self.static_obst_radius_mean))

            #Ensure that the obstacle is not too close to the path
            # while np.linalg.norm(self.path(self.path.get_closest_arclength(obstacle.position)) - obstacle.position) < (obstacle.radius + min_distance_to_path):
            #     obstacle = CircularObstacle(*helpers.generate_obstacle(self.rng, self.path, self.vessel, displacement_dist_std=self.displacement_dist_std))
            self.obstacles.append(obstacle)

        self._update()

        #Activate safety filter
        if safety_filter_rank != -1:
            self.vessel.safety_filter = safety_filter
            self.vessel.activate_safety_filter(self, safety_filter_rank)
            

