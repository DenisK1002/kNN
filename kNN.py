"""
Implementation of the k-Nearest-Neighbour algorithm
"""

from utils import *
import sys

class kNN():
    def __init__(self, dataset_filename = "", k = 1, distance_method = "euclidean"):
        self.dataset_filename = dataset_filename
        self.model = None
        self.distance_method = distance_method
        self.k = k
        self.check_parameters()
        self.fit()

    def check_parameters(self):
        """
        Checks if input parameters are valid.
        Exits if invalid.
        """
        valid_dataset_filename = self.dataset_filename.split(".")[-1] == "csv"
        valid_k = type(self.k) == int and self.k > 0
        valid_distance_method = self.distance_method in DistanceMethod.get_methods()
        if not (valid_dataset_filename and valid_k and valid_distance_method):
            print("One or more arguments invalid. Check them!")
            exit(1)

    def evaluate(self, test_data):
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
        print(f"\nTraining kNN...", end="")
        dataset = Dataset(self.dataset_filename)
        dataset_train, dataset_test = dataset.split_train_test()
        self.model = dataset_train
        print(" Done.")

        return self.evaluate(dataset_test)

    def predict(self, datapoint, k):
        """
        Classifies datapoint to a class using 
        """
        neighbours = self.model.k_Neighbours(datapoint, k, self.distance_method)
        count_classes = {}
        for n in neighbours:
            if n.category in count_classes.keys():
                count_classes[n.category] = count_classes[n.category] + 1
            else:
                count_classes.update({n.category: 1})
        
        # return the class with the highest occurence in respect to datapoints neighbours
        return list(sorted(count_classes.items(), key=lambda c: c[1], reverse=True))[0][0]




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("3 additional Parameters needed:")
        print("filename, k, distance_method")

    kNN(sys.argv[1])
