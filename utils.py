import math
import random

class Dataset():
    def __init__(self, dataset_filename, max_size=1000):
        self.dataset_filename = dataset_filename

        # max size of dataset
        self.max_size = max_size

        self.datapoints = self.init_datapoints()

    def split_train_test(self, ratio=0.8):
        """
        Splits the dataset into a train and a test set.
        Default ratio is 80% train 20% test.
        """

        random_indizes = random.choices(range(0, len(self.datapoints)), k=int(len(self.datapoints)*(1-ratio)))
        # for _ in range(int(len(self.datapoints)*1-ratio)):
        #     random_indizes.append(random.randint(0, len(self.datapoints)))
        # random_indizes = [random.randint(0, len(self.datapoints)) for _ in range()]
        # test = random.choices(self.datapoints, k=int(len(self.datapoints)*(1-ratio)))
        # test = [
        #     self.datapoints[random.randint(0, len(self.datapoints))]    # random datapoints
        #     for _ in range(0, len(self.datapoints)*(1-ratio))
        # ]
        test = [self.datapoints[i] for i in random_indizes]
        random_indizes.sort(reverse=True)
        for i in random_indizes:
            self.datapoints.pop(i)
        
        train = Dataset(self.dataset_filename, self.max_size)
        train.datapoints = self.datapoints
        return train, test

    def init_datapoints(self) -> list:
        """
        Reading csv file and creating datapoints.
        Returns list of [DataPoints]
        """
        datapoints = []
        with open(self.dataset_filename, "r") as csv:
            header = csv.readline().split(",")

            for _ in range(self.max_size):
                row = csv.readline().rstrip("\n")
                if row == "":
                    continue
                point = row.split(",")
                features = list(map(float, point[:-1]))
                category = int(point[-1])
                datapoints.append(DataPoint(features, category, header))
                
        return datapoints
    
    def k_Neighbours(self, datapoint, k):
        """
        Returns k closest neighbours to datapoint in ascending order
        """
        neighbours = list(sorted(self.datapoints, key=lambda d: datapoint.distance(d, DistanceMethod.euclidian)))
        if neighbours[0] == datapoint:
            neighbours.pop(0)

        return neighbours

class DistanceMethod:
    manhattan = "manhattan"
    euclidian = "euclidian"

class DataPoint():
    """
    Class Representing a Datapoint in the dataset.
    Consisting of its features and the according categorie the datapoint belongs to.
    """
    def __init__(self, features: list, category: int, category_identifier: list):
        self.features = features
        self.category = category
        self.category_identifier = []

    def euclidian_distance(self, datapoint_2) -> float:
        """
        Calculates the euclidian distance to another point.
        """
        return math.sqrt(sum([pow((p-q), 2) for p, q in zip(self.features, datapoint_2.features)]))

    def manhattan_distance(self, datapoint_2) -> float:
        """
        Calculates the euclidian distance to another point.
        """
        return sum([abs(p - q) for p, q in zip(self.features, datapoint_2.features)])

    def distance(self, datapoint_2, distance_method: DistanceMethod):
        """
        Returns distance [Manhattan or euclidian] to another point.
        """

        if distance_method == DistanceMethod.euclidian:
            return self.euclidian_distance(datapoint_2)
        elif distance_method == DistanceMethod.manhattan:
            return self.manhattan_distance(datapoint_2)
        else:
            raise ValueError(distance_method)
    
    def __eq__(self, __o: object) -> bool:
        return all([f == f__o for f, f__o in zip(self.features, __o.features)]) and all([c == c__o for c, c__o in zip(self.category, __o.category)])
    
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)

    def __repr__(self) -> str:
        return f"{self.features} | {self.category}"