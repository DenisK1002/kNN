# k-Nearest-Neighbour Classifier Algorithm
This was implemented as a part of my studies at OTH Regensburg. <br/>
Sake of the program is to test different datasets on the kNN algorithm and evaluate its performance.

## Usage
It can be used entirely by command-line <br/>
Example <br/>
`python kNN.py iris.csv 1 euclidean`

Arguments:
1. path to dataset as csv file
2. the amount of neighbours a datapoint needs to consider
3. the distance calculation method. There is euclidean and manhattan as valid distance arguments.

The default parameters are "", 1, "euclidean"

## kNN-Implementation
Executing the python program will run the kNN algorithm on specified arguments. <br/>
It first will load the dataset into memory and split it into a train and test set by a ration of 80:20.
Afterwards an evaluation on the test data is done.
The algorithm will predict the class of each of the datapoint in the test data and calculates how often it correctly
classified a datapoint.

Example output for: <br/>
`python kNN.py agk-ue08-p1.csv 1 euclidean`
```
Training kNN... Done.
Evaluating Model... Done.

----Results----

Predicted 1 out of 1 correctly
Test error rate: 0.0
```

Since in the agk-ue08-p1.csv are 5 datapoints, 4 of them are used for training and 1 is used for testing.

### Splitting Dataset into train and test data
As mentioned the dataset is split into train and test data in the [split_train_test()](/utils.py?line=20) function in the utils.py utility file.
As you may observe the selection of test data is random for the sake of avoiding repetition when executing the algorithm with the same parameters multiple times.