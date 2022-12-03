# k-Nearest-Neighbour Classifier Algorithm
This was implemented as a part of my studies at OTH Regensburg. <br/>
Sake of the program is to test different datasets on the kNN algorithm and evaluate its performance.

## Usage
It can be used entirely by command-line <br/>
Example <br/>
`python kNN.py 150 iris.csv 1 euclidean`

Arguments:
1. max size of datapoints to be read
2. path to dataset as csv file
3. the amount of neighbours a datapoint needs to consider
4. the distance calculation method. There is euclidean and manhattan as valid distance arguments.

The default parameters are dataset_filename="", k=1, distance_method="euclidean".
Max-size argument needs to be provided

## kNN-Implementation


Example output for: <br/>
`python kNN.py 6 agk-ue08-p1.csv 1 euclidean`
```
Loading Dataset... Done.
5 / 6 correctly classified (0.8333)
```

### Leave-one-out experiment
The [loo experiment](kNN.py#L76) will iterate over all points in the dataset and assign them individually as the test datapoint.
The other n-1 datapoints are used for training.
The result of the predictions of all test datapoints is accumulated and printed out to the terminal.