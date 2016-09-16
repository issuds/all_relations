# Data-Driven, Statistical Learning Method for Automatically Extracting Path Models

This code can be used to determine weights for every pair of relations between given set of concepts which describe the strength of relation.

## Data format

See examples in "dataset" folder. Accepted data format is csv file, which has columns in the following format:

conceptA1, conceptA2, ... conceptAN, conceptB1, conceptB2, ... conceptBN, 

where A, B, ... are names of concepts, and by numbers 1, 2, ... are denoted features for a given concept.

## Dependencies

To run this code, numpy and sklearn python packages are necessary. The easiest way to set up all necessary dependencies is to install [Anaconda python distribution](https://www.continuum.io/downloads)

Additionally, [tensorflow](https://www.tensorflow.org/) is necessary if you wish to use artificial neural networks. Currently (September 2016) installation of tensorflow is only supported on Linux based OS (e.g. Ubuntu). 

## Running the code

Python script "main.py" can be used to run the code with your own data, provided that the data is in the compatible format. See "main.py" for further details. Simply run the script after you checked out the repository to run the code on the data used in our paper.



