# Data-Driven, Statistical Learning Method for Automatically Extracting Path Models

This code can be used to determine weights for every pair of relations between given set of concepts which describe the strength of relation.

## Installation and dependencies

We offer two ways to install and use the code: using docker container or via installation of all dependencies on your system.
If you are not too familiar with python and pycharm or if you use Windows, it is advised that you use docker based installation. 

### Installation using docker

Please follow the instructions here:
https://hub.docker.com/r/ed3s/all-relations/

### Installation on your system

Please note: installation that can use artificial neural networks is not possible on Windows, as currently [tensorflow](https://www.tensorflow.org/) necessary to run such ann currently (September 2016) does not support Windows. 

1. Install [Anaconda python distribution](https://www.continuum.io/downloads).
2. Open terminal in Anaconda Navigator and update conda: condo update conda
3. Create New Environment with a name "condaenv" (choose your name), Python option and version 2.7; import package pandas 
4. Install [Tensorflow](https://www.tensorflow.org/): Open terminal in Anaconda Navigator, and type 
`conda install -c jjhelmus tensorflow=0.10.0rc0` (`--name condaenv` if you install in a specific environment)
If you are on Windows, you can skip this step, however ANNs  will not work.
5. Clone or download this repository in some folder. 
6. Install [PyCharm IDE](https://www.jetbrains.com/pycharm/)
7. Open PyCharm IDE. Go to "File > Open" and choose to open the folder with source code from repository. 
Open File>Default Settings>Project Interpreter ; select project interpreter by the environment name "condaenv"

## Running the code

Python script "main.py" can be used to run the code with your own data, provided that the data is in the compatible format. See "main.py" for further details. Simply run the script after you checked out the repository to run the code on the data used in our paper.

## Data format for your own datasets

See examples in "dataset" folder. Accepted data format is csv file, which has columns in the following format:

A1, A2, ... AN, B1, B2, ... BN, C1 ...

where A, B, C ... are names of concepts, and by numbers 1, 2, ... are denoted features for a given concept.

For example, for two concepts A and B, where A has 2 features, and B has 3 features, the dataset csv would look as follows:

| A1 | A2 | B1 | B2 | B3 |
|----|----|----|----|----|
| 0  | 1  | 0  | 1  | 2  |
| 3  | 0  | 2  | 3  | 0  |
| 2  | 0  | 2  | 1  | 2  |
| 1  | 3  | 2  | 0  | 1  |
| 2  | 3  | 0  | 1  | 2  |

## Structure of results

Results are stored in the `results_folder` specified in the main.py. The structure of results is as follows: for every class of predictive models considered, the file "\[datasetname\]\_\[model\_class\].csv" is created, where the contents of csv file contain matrix with weights of strength relation between every pair of relations. 

In addition, `matrix.csv` contains average values of strength in matrix format, and ranking.csv contains list of one to one relations sorted by average relation strength. The file `for_evaluation.csv` contains data that was not used for the relation strength extraction and can be used for further verification of extracted model. The file `dataset.csv` contains original dataset. 
