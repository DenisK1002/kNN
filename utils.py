"""
Utility file
"""

import math
import random
from copy import deepcopy

class DistanceMethod:
    manhattan = "manhattan"
    euclidean = "euclidean"

    def get_methods() -> list:
        return ["manhattan", "euclidean"]

class Dataset():
    def __init__(self, dataset_filename, max_size=160):
        self.dataset_filename = dataset_filename

        # max size of dataset
        self.max_size = max_size

        self.datapoints = self.init_datapoints()
        
        self.header = []

    def leave_one_out(self, index):
        """
        Splits the dataset into 1 test datapoint and n-1 train datapoints
        """
        test = self.datapoints[index]

        # copy dataset without test datapoint
        train = Dataset(self.dataset_filename, self.max_size)
        train.datapoints = deepcopy(self.datapoints)
        train.datapoints.pop(index)
        train.header = self.header

        return train, test

    def split_train_test(self, ratio=0.8):
        """
        Splits the dataset into a train and a test set. (cross-validation)
        Default ratio is 80% train 20% test.
        """
        # splits dataset randomly by ratio
        random_indizes = random.choices(range(0, len(self.datapoints)-1), k=int(len(self.datapoints)*(1-ratio)))
        test = [self.datapoints[i] for i in random_indizes]
        random_indizes.sort(reverse=True)
        for i in random_indizes:    
            self.datapoints.pop(i)
        
        return self, test

    def init_datapoints(self) -> list:
        """
        Reading csv file and creating datapoints.
        Returns list of [DataPoints]
        """
        datapoints = []
        with open(self.dataset_filename, "r") as csv:
            self.header = csv.readline().split(",")
            
            for _ in range(self.max_size):
                row = csv.readline().rstrip("\n")
                if row == "":
                    continue
                point = row.split(",")
                features = list(map(float, point[:-1]))
                category = int(point[-1])
                datapoints.append(DataPoint(features, category))
                
        return datapoints
    
    def k_Neighbours(self, datapoint, k, distance_method=DistanceMethod.euclidean) -> list:
        """
        Returns k closest neighbours to datapoint in ascending order
        """
        neighbours = list(sorted(self.datapoints, key=lambda d: datapoint.distance(d, distance_method)))
        if neighbours[0] == datapoint:
            neighbours.pop(0)

        return neighbours[:k]

    def __len__(self) -> int:
        return len(self.datapoints)

class DataPoint():
    """
    Class Representing a Datapoint in the dataset.
    Consisting of its features and the according category the datapoint belongs to.
    """
    def __init__(self, features: list, category: int):
        self.features = features
        self.category = category

    def euclidean_distance(self, datapoint_2) -> float:
        """
        Calculates the euclidean distance to another point.
        """
        return math.sqrt(sum([pow((p-q), 2) for p, q in zip(self.features, datapoint_2.features)]))

    def manhattan_distance(self, datapoint_2) -> float:
        """
        Calculates the manhattan distance to another point.
        """
        return sum([abs(p - q) for p, q in zip(self.features, datapoint_2.features)])

    def distance(self, datapoint_2, distance_method: DistanceMethod) -> float:
        """
        Returns distance [Manhattan or euclidean] to another point.
        """
        if distance_method == DistanceMethod.euclidean:
            return self.euclidean_distance(datapoint_2)
        elif distance_method == DistanceMethod.manhattan:
            return self.manhattan_distance(datapoint_2)
        else:
            raise ValueError(distance_method)

    def __repr__(self) -> str:
        return f"{self.features} | {self.category}"
