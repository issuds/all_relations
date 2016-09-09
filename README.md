# Data-Driven, Statistical Learning Method for Automatically Extracting Path Models

This code can be used to determine weights for every pair of relations between given set of concepts which describe the strength of relation.

## Data format

See examples in "dataset" folder.
Accepted data format is csv file, which has columns in the following format:

conceptA1, conceptA2, ... conceptAN, conceptB1, conceptB2, ... conceptBN, 

where A, B, ... are names of concepts, and by numbers 1, 2, ... are denoted features for a given concept.

For further details on the data format refer to "dataset_parser.py".

## Installation

The most certain way to install everything is to use Ubuntu 15 virtual machine. To create one, you require virtualbox to be installed:

https://www.virtualbox.org/wiki/Downloads

Install Ubuntu 15 using VirtualBox.

After installation of Ubuntu, run the following code in terminal (ctrl alt t):

Install python dependencies:

sudo apt-get install python-pip -y

sudo pip install numpy

sudo apt-get install python-scipy -y

sudo apt-get install python-scipy python-dev python-pip python-nose g++ libopenblas-dev git -y

Install Pycharm IDE:

sudo add-apt-repository ppa:ubuntu-desktop/ubuntu-make

sudo apt-get update

sudo apt-get install ubuntu-make -y

umake ide pycharm

The most straightforward way to share data with virtual machine is through cloud data storage (e.g. dropbox, google drive).

## Running the code

Clone this repository in some folder. 

main_script.py is the main script that can be used to reproduce our results. All the data we use is stored in "datasets" folder in the pycharm project.
By default, the results of computation are cached in the files '*.bin'. If you make changes to the code, rename this files or delete them so that results are recomputed.


