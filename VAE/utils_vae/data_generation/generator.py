from radar import RadarEnvironment
import csv
from tqdm import tqdm
import numpy as np


class DataGenerator():
    def __init__(self, num_circles:int, num_moving:int, write_path:str, num_obs:int) -> None:
        self.write_path = write_path
        self.num_circles = num_circles
        self.num_moving_obstacles = num_moving
        self.num_obs = num_obs
    
    def generate_data(self):
        with open(self.write_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            for _ in tqdm(range(self.num_obs), desc="Generating dataset"):
                # Create new random radar observation for each line
                r = RadarEnvironment(num_circles=self.num_circles, num_moving_obstacles=self.num_moving_obstacles)
                r.generate_scenario()
                sensor_values = r.get_sensor_values()
                writer.writerow(sensor_values)
                csvfile.flush()  # Flush the buffer to ensure data is written immediately
                del r # delete radar object
        print(f"Dataset saved to {self.write_path}")


def concat_csvs(csvs:tuple, output_path:str):
   """Stacks n csvs from list csvs containing paths into one csv at output_path"""
   arrs = tuple([np.loadtxt(csv) for csv in csvs])
   arr = np.concatenate(arrs, axis=0)
   np.savetxt(output_path, arr, delimiter=" ")


if __name__ == "__main__":
    
    # Generate dataset with only moving obstacles 
    n_moving_dense = 2500
    path_moving_dense = "../../data/LiDAR_synthetic_onlyMovingObst_dense.csv"
    data_generator_moving_dense = DataGenerator(num_circles=0, num_moving=5, num_obs=n_moving_dense, write_path=path_moving_dense)
    data_generator_moving_dense.generate_data()

    # Generate dataset with only static obstacles 
    n_static_dense = 2500
    path_static_dense = "../../data/LiDAR_synthetic_onlyStaticObst_dense.csv"
    data_generator_static_dense = DataGenerator(num_circles=5, num_moving=0, num_obs=n_static_dense, write_path=path_static_dense)
    data_generator_static_dense.generate_data()

    # Generate dataset with both static and moving obstacles 
    n_static_moving = 5000
    path_static_moving = "../../data/LiDAR_synthetic_staticMovingObst.csv"
    data_generator_static_moving = DataGenerator(num_circles=3, num_moving=2, num_obs=n_static_moving, write_path=path_static_moving)
    data_generator_static_moving.generate_data()
    
    """
    # Generate dataset with only moving obstacles (SPARSE)
    n_moving_sparse = 1000
    path_moving_sparse = "../../data/LiDAR_synthetic_onlyMovingObst_sparse.csv"
    data_generator_moving_sparse = DataGenerator(num_circles=0, num_moving=3, num_obs=n_moving_sparse, write_path=path_moving_sparse)
    data_generator_moving_sparse.generate_data()

    # Generate dataset with only static obstacles (SPARSE)
    n_static_sparse = 1000
    path_static_sparse = "../../data/LiDAR_synthetic_onlyStaticObst_sparse.csv"
    data_generator_static_sparse = DataGenerator(num_circles=3, num_moving=0, num_obs=n_static_sparse, write_path=path_static_sparse)
    data_generator_static_sparse.generate_data()

    # Generate empty dataset
    n_empty = 1000
    path_empty = "../../data/LiDAR_synthetic_empty.csv"
    data_generator_empty = DataGenerator(num_circles=0, num_moving=0, num_obs=n_empty, write_path=path_empty)
    data_generator_empty.generate_data()"""

