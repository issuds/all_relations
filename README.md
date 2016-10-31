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

To run this code, numpy and sklearn python packages are necessary. The easiest way to set up all necessary dependencies is to install [Anaconda python distribution](https://www.continuum.io/downloads)

## Running the code

Python script "main.py" can be used to run the code with your own data, provided that the data is in the compatible format. See "main.py" for further details. Simply run the script after you checked out the repository to run the code on the data used in our paper.

## Data format for your own datasets

See examples in "dataset" folder. Accepted data format is csv file, which has columns in the following format:

conceptA1, conceptA2, ... conceptAN, conceptB1, conceptB2, ... conceptBN, 

where A, B, ... are names of concepts, and by numbers 1, 2, ... are denoted features for a given concept.


