import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, LineString, MultiPolygon
from shapely.ops import unary_union
import shapely.geometry
import shapely.affinity



# Created by Henrik S. Berg (Oct 2023) with inspiration from Meyer (2020)
# Tailored for this project by Eirik R. Barlaug (Nov 2023)


class RadarEnvironment():
    """Radar environment for generating synthetic data in the same manner as in the RL environment."""
    def __init__(self, sensor_range=150, num_sensors=180, num_circles=10, mean_radius = 25, num_moving_obstacles = 10, moving_mean = 10, safe_zone_radius = 10 ):
        self.sensor_range = sensor_range
        self.num_sensors = num_sensors
        self.num_circles = num_circles
        #self.circle_radius = circle_radius
        self.mean_radius = mean_radius
        self.main_object = Point(0, 0)
        self.safe_zone_radius = safe_zone_radius
        self.circles = []
        self.moving_obstacles = []
        self.num_moving_obstacles = num_moving_obstacles
        self.moving_mean = moving_mean
        self.sensor_distances = np.zeros(self.num_sensors)
        self.generate_scenario()
        self.all_circles = unary_union(self.circles) 
        
    
    def movingobstacle(self,width,heading):
        points = [
            (-width/2, -width/2),
            (-width/2, width/2),
            (width/2, width/2),
            (3/2*width, 0),
            (width/2, -width/2),
        ]
        boundary_temp = shapely.geometry.Polygon(points)
        boundary = shapely.affinity.rotate(boundary_temp, heading, use_radians=True, origin='centroid')
        return boundary

    
    def generate_moving_obstacles(self):
        obstacles = []
        for _ in range(self.num_moving_obstacles):
            width = np.random.poisson(self.moving_mean)
            heading = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
            # Generate a random position for the obstacle
            pos_x = np.random.uniform(-self.sensor_range, self.sensor_range)
            pos_y = np.random.uniform(-self.sensor_range, self.sensor_range)
            obstacle = self.movingobstacle(width, heading)
            # Move the obstacle to the random position
            moved_obstacle = shapely.affinity.translate(obstacle, pos_x, pos_y)
            obstacles.append(moved_obstacle)
        return obstacles



    def generate_scenario(self):
        # Define a safe zone around the main object
        safe_zone_radius = self.safe_zone_radius  # This should be defined in __init__
        safe_zone = self.main_object.buffer(safe_zone_radius)

        # Initialize an empty list for moving obstacles
        self.moving_obstacles = []

        # Keep track of all obstacles for intersection checks
        all_obstacles = unary_union(self.circles)

        # Generate random circles that do not intersect with the safe zone
        while len(self.circles) < self.num_circles:
            radius = np.random.poisson(self.mean_radius)
            pos_x = np.random.uniform(-self.sensor_range, self.sensor_range)
            pos_y = np.random.uniform(-self.sensor_range, self.sensor_range)
            circle = Point(pos_x, pos_y).buffer(radius)
            if not circle.intersects(safe_zone) and not circle.intersects(all_obstacles):
                self.circles.append(circle)
                all_obstacles = unary_union([all_obstacles, circle])

        # Generate moving obstacles that do not intersect with circles or the safe zone
        while len(self.moving_obstacles) < self.num_moving_obstacles:
            width = np.random.poisson(self.moving_mean)
            heading = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
            pos_x = np.random.uniform(-self.sensor_range, self.sensor_range)
            pos_y = np.random.uniform(-self.sensor_range, self.sensor_range)
            obstacle = self.movingobstacle(width, heading)
            moved_obstacle = shapely.affinity.translate(obstacle, pos_x, pos_y)
            if not moved_obstacle.intersects(safe_zone) and not moved_obstacle.intersects(all_obstacles):
                self.moving_obstacles.append(moved_obstacle)
                all_obstacles = unary_union([all_obstacles, moved_obstacle])

        # Combine static and moving obstacles into one shape for sensor detection
        self.all_obstacles = all_obstacles

        # Calculate sensor distances
        angles = np.linspace(0, 2 * np.pi, self.num_sensors, endpoint=False) - np.pi / 2
        self.sensor_distances = np.full(self.num_sensors, self.sensor_range, dtype=float)  # Initialize with max range

        for i, angle in enumerate(angles):
            # Calculate end point of the ray based on the sensor range and angle
            end_x = self.main_object.x + self.sensor_range * np.cos(angle)
            end_y = self.main_object.y + self.sensor_range * np.sin(angle)
            ray = LineString([(self.main_object.x, self.main_object.y), (end_x, end_y)])
            
            # Find intersection with all obstacles
            intersection = ray.intersection(self.all_obstacles)
            
            if intersection.is_empty:
                continue  # No intersection, keep the sensor range as the distance
            
            # If the intersection is a point or a collection of points
            if 'Point' in intersection.geom_type:
                # Handle the case where intersection can be MultiPoint
                if intersection.geom_type == 'MultiPoint':
                    closest_point = min(intersection, key=lambda p: self.main_object.distance(p))
                    self.sensor_distances[i] = self.main_object.distance(closest_point)
                else:
                    self.sensor_distances[i] = self.main_object.distance(intersection)
            # If the intersection is a segment (LineString) or a collection of segments
            elif 'LineString' in intersection.geom_type:
                # Handle the case where intersection can be MultiLineString
                if intersection.geom_type == 'MultiLineString':
                    # Find the closest point in all the line segments
                    closest_point_distance = self.sensor_range
                    for line in intersection.geoms:  # Use .geoms to iterate over the line segments
                        first_point = line.coords[0]
                        distance = self.main_object.distance(Point(first_point))
                        if distance < closest_point_distance:
                            closest_point_distance = distance
                    self.sensor_distances[i] = closest_point_distance
                else:
                    first_point = intersection.coords[0]
                    self.sensor_distances[i] = self.main_object.distance(Point(first_point))

    def get_sensor_values(self):
        return self.sensor_distances
    
    def plot_scenario(self):
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-self.sensor_range, self.sensor_range)
        ax.set_ylim(-self.sensor_range, self.sensor_range)
        ax.set_aspect('equal')

        # Plot circles
        for circle in self.circles:
            bounds = circle.bounds
            radius = (bounds[2] - bounds[0]) / 2
            ax.add_patch(patches.Circle((circle.centroid.x, circle.centroid.y), radius, color='blue', alpha=0.5))

        # Plot moving obstacles
        for obstacle in self.moving_obstacles:
            x, y = obstacle.exterior.xy
            ax.plot(x, y, 'orange', alpha=0.7)

        # Plot main object
        ax.plot(self.main_object.x, self.main_object.y, 'ro')  # Main object as a red dot

        angles = np.linspace(0, 2 * np.pi, self.num_sensors, endpoint=False)
        for i, angle in enumerate(angles):
            # Calculate end point of the ray based on the sensor range and angle
            end_x = self.main_object.x + self.sensor_range * np.cos(angle)
            end_y = self.main_object.y + self.sensor_range * np.sin(angle)
            ray = LineString([(self.main_object.x, self.main_object.y), (end_x, end_y)])
            
            # Find intersection with all obstacles
            intersection = ray.intersection(self.all_obstacles)
            
            if intersection.is_empty:
                # If no intersection, draw the ray in green
                ax.plot([self.main_object.x, end_x], [self.main_object.y, end_y], 'g-')
            else:
                # Find the closest intersection point to the sensor
                closest_point = None
                min_dist = float('inf')
                if 'Multi' in intersection.geom_type:
                    for geom in intersection.geoms:
                        if geom.geom_type == 'Point':
                            dist = self.main_object.distance(geom)
                            if dist < min_dist:
                                min_dist = dist
                                closest_point = geom
                        elif geom.geom_type == 'LineString':
                            dist = self.main_object.distance(Point(geom.coords[0]))
                            if dist < min_dist:
                                min_dist = dist
                                closest_point = Point(geom.coords[0])
                else:
                    if intersection.geom_type == 'Point':
                        closest_point = intersection
                    elif intersection.geom_type == 'LineString':
                        closest_point = Point(intersection.coords[0])

                # Draw the closest intersection point
                if closest_point:
                    ax.plot([self.main_object.x, closest_point.x], [self.main_object.y, closest_point.y], 'r-')

        #plt.show()
        plt.savefig("test.png")



radar_env = RadarEnvironment(num_circles=4, num_moving_obstacles=7)
radar_env.plot_scenario()
sensor_values = radar_env.get_sensor_values()
print(sensor_values)

