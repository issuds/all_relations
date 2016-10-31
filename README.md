# Data-Driven, Statistical Learning Method for Automatically Extracting Path Models

This code can be used to determine weights for every pair of relations between given set of concepts which describe the strength of relation.

## Installation and dependencies

We offer two ways to install and use the code: using docker container or via installation of all dependencies on your system.
If you are not too familiar with python and pycharm or if you use Windows, it is advised that you use docker based installation. 

### Installation using docker

Please follow the instructions here:
https://hub.docker.com/r/ed3s/all-relations/

### Installation on your system

Please note: installation that can use artificial neural networks is not possible on Windows, as currently [tensorflow](https://www.tensorflow.org/) necessary to run such ann as of now (September 2016) does not support Windows. 

1. Install [Anaconda python distribution](https://www.continuum.io/downloads)
2. Install [Tensorflow](https://www.tensorflow.org/): Open terminal, and type 
`conda install -c jjhelmus tensorflow=0.10.0rc0`
If you are on Windows, you can skip this step, however ANN option will not work.
3. Clone or download this repository in some folder. 
4. Install [PyCharm IDE](https://www.jetbrains.com/pycharm/download/downloads)
5. Open PyCharm IDE. Go to "File > Open" and choose to open the folder with unpacked source code from repository. 
6. You might need to configure the interpreter for PyCharm. Select anacoda python 2.7 interpreter by going to file > settings > project > interpreter.

## Running the code

Python script "main.py" can be used to run the code with your own data, provided that the data is in the compatible format. See "main.py" for further details. Simply run the script after you checked out the repository to run the code on the data used in our paper.

## Data format for your own datasets

See examples in "dataset" folder. Accepted data format is csv file, which has columns in the following format:

conceptA1, conceptA2, ... conceptAN, conceptB1, conceptB2, ... conceptBN, 

where A, B, ... are names of concepts, and by numbers 1, 2, ... are denoted features for a given concept.


