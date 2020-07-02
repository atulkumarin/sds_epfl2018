# sds_epfl2018

This repository contains work in progress towards a project for the course Systems for Data Science taken at EPFL.

The project is about implementing a distributed stochastic gradient descent (SGD) for training support vector machine (SVM)
classifiers for a multi-class classification problem.
## How to run
- The script in docker/run.sh starts the whole training pipeline.
- The Kubernetes parameters and the training parameters can be changed throught the json file in docker/config.json.
- The parameters description can be found in kubernetes/config_template.yaml.

## Assumptions :
- The svm\_server.py script needs to be started before svm\_client.py script.
- The co-ordinator is a grpc client and the workers are grpc servers in our implementation.
- The default number of workers is 3 . In case you need to change it, pass the number of workers as an argument in the command line (Use the same number for both server and client script). 
- To stop the training (preferably after 100 iterations), use a keyboard interrupt (ctrl+c)
- The default learning rate, regularization and batch size per worker is 0.1, 0 and 80 respectively. To change this values, modify lr, reg\_factor and nb\_batches variables in svm\_client.py.

1. **Memory:**

	We assume that the whole dataset cannot fit in memory at the workers. The co-ordinator is responsible for building the index that associates data points/samples to their corresponding seek positions in the file before starting the training process. The indexes required for the current iteration by a particular worker are sent as a part of the grpc request message.
	
	We could have got rid of the indexing part and sample data points at the workers but however this could imply that different workers can work on the same data point in one iteration.
	
	Hence, we load only the batch that is needed for the current iteration from the disk.

2. **Validation Loss:**

	We are computing the validation loss using one of the test files (truncated) during the training process. This task is also splitter among the workers in order to get faster computation.

3. **Building the training set:**

	Since the original data is multi-label multi-class, we split it into two classes (negative and positive examples) such that we have approximately equal number (balanced) of positive and negative examples. (Refer to the python notebook file balanced_class.ipynb). The created datasets are 'labels\_balanced.dat' and 'test\_labels\_balanced.dat'


