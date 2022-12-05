"""
Implementation of the k-Nearest-Neighbour algorithm
"""

from utils import *
import sys

class kNN():
    def __init__(self, max_size, dataset_filename = "", k = 1, distance_method = "euclidean"):
        self.max_size = int(max_size)
        self.dataset_filename = dataset_filename
        self.dataset = None
        self.distance_method = distance_method
        self.k = int(k)
        self.check_parameters()
        self.fit()

    def check_parameters(self):
        """
        Checks if input parameters are valid.
        Exits if invalid.
        """
        valid_size = self.max_size > 0
        valid_dataset_filename = self.dataset_filename.split(".")[-1] == "csv"
        valid_k = type(self.k) == int and self.k > 0
        valid_distance_method = self.distance_method in DistanceMethod.get_methods()
        if not (valid_size and valid_dataset_filename and valid_k and valid_distance_method):
            print("One or more arguments invalid. Check them!")
            exit(1)

    def evaluate(self, test_data) -> float:
        """
        Evaluates kNN using test error 
        """
        print("Evaluating Model...", end="")
        correctly_classified = 0
        for datapoint in test_data:
            predicted_class = self.predict(datapoint, self.k)
            if predicted_class == datapoint.category:
                correctly_classified += 1
        print(" Done.")

        evaluation = (len(test_data)-correctly_classified)/len(test_data)
        print("\n----Results----\n")
        print(f"Predicted {correctly_classified} out of {len(test_data)} correctly")
        print(f"Test error rate: {evaluation}")

        return evaluation

    def fit(self):
        """
        Fits the training data to the model.
        """
        print(f"Loading Datatset...", end="")
        self.dataset = Dataset(self.dataset_filename, self.max_size)
        print(" Done.")
        
        self.leave_one_out_experiment()

    def predict(self, train, datapoint, k) -> int:
        """
        Classifies datapoint to a class using model
        """
        neighbours = train.k_Neighbours(datapoint, k, self.distance_method)
        count_classes = {}
        for n in neighbours:
            if n.category in count_classes.keys():
                count_classes[n.category] = count_classes[n.category] + 1
            else:
                count_classes.update({n.category: 1})
        
        # return the class with the highest occurence in respect to datapoints neighbours
        return list(sorted(count_classes.items(), key=lambda c: c[1], reverse=True))[0][0]
    
    @measure_runtime
    def leave_one_out_experiment(self):
        """
        Performs leave one out experiment over all datapoints in the dataset.
        It tests the classification prediction for each datapoint with given k.
        """
        correctly_classified = 0
        for i in range(len(self.dataset)):
            train, test = self.dataset.leave_one_out(i)
            if self.predict(train, test, self.k) == test.category:
                correctly_classified += 1
        
        percentage = correctly_classified / len(self.dataset)
        print(f"{correctly_classified} / {len(self.dataset)} correctly classified ({percentage:.4f})")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("4 additional Parameters needed:")
        print("max-size, filename, k, distance_method")

    kNN(*sys.argv[1:])
