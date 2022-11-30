"""
Implementation of the k-Nearest-Neighbour algorithm
"""

from utils import *
import sys

class kNN():
    def __init__(self, dataset_filename = ""):
        self.dataset_filename = dataset_filename
        self.model = None

    def evaluate(self, test_data):
        """
        Evaluates kNN using test error 
        """
        print("Evaluating Model...", end="")
        correctly_classified = 0
        for datapoint in test_data:
            predicted_class = self.predict(datapoint)
            if predicted_class == datapoint.category:
                correctly_classified += 1
        print(" Done.")

        print("\n----Results----\n")
        print(f"Predicted {correctly_classified} out of {len(test_data)} correctly")
        print(f"Test error rate: {(len(test_data)-correctly_classified)/len(test_data)}")

    def fit(self):
        """
        Fits the training data to the model.
        """
        print(f"\nTraining kNN...", end="")
        dataset = Dataset(self.dataset_filename)
        dataset_train, dataset_test = dataset.split_train_test()
        self.model = dataset_train
        print(" Done.")

        self.evaluate(dataset_test)

    def predict(self, datapoint, k=1):
        """
        Classifies datapoint to a class using 
        """
        neighbours = self.model.k_Neighbours(datapoint, k)
        count_classes = {}
        for n in neighbours:
            if n.category in count_classes.keys():
                count_classes[n.category] = count_classes[n.category] + 1
            else:
                count_classes.update({n.category: 1})
        
        # return the class with the highest occurence in respect to datapoints neighbours
        return list(sorted(count_classes.items(), key=lambda c: c[1], reverse=True))[0][0]




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please enter a filename to a csv document")

    kNN(sys.argv[1])
