# Automated extraction of SEM structure

This software is a supplement to following publications: 

* Maass, W. & Shcherbatyi, I. Inductive Discovery By Machine Learning for Identification of Structural Models, The 37th International Conference on Conceptual Modeling (ER), 2018. 

* Maass, W. & Shcherbatyi, I. Data-Driven, Statistical Learning Method for Inductive Confirmation of Structural Models, Hawaii International Conference on System Sciences (HICSS), 2017. 

### Requirements

Python 3 is recommended to run the code. 
If something does not work on python 2 leat us know. 
You require the following packages to be installed on your system: 
`numpy`, `scipy`, `pandas` `scikit-learn`, `scikit-optimize`, `tqdm`. 

### Installation on Ubuntu

Open a terminal. In the terminal, navigate to the folder of the repository.
When in the folder, run following commands:

```bash
bash install.sh
```

If you also wish to install Pycharm IDE to use as GUI python editor, run the 
following command in the folder of the repository:
```bash
bash pycharm.sh
```

### Installation on MacOS

1. Install [Anaconda python distribution](https://www.continuum.io/downloads).
2. Open terminal in Anaconda Navigator and update conda: condo update conda
3. Create New Environment with a name "condaenv" (choose your name), Python option and version 2.7; import package pandas 
5. Clone or download this repository in some folder. 
6. Install [PyCharm IDE](https://www.jetbrains.com/pycharm/)
7. Open PyCharm IDE. Go to "File > Open" and choose to open the folder with source code from repository. 
Open File>Default Settings>Project Interpreter ; select project interpreter by the environment name "condaenv"

## Running the code

Python script "main.py" can be used to run the code with your own data, provided that the data is in the compatible format. See "main.py" for further details. Simply run the script after you checked out the repository to run the code on the data used in our paper.

## Structure of results

Results are stored in the `results` folder. You can load and print the results using the `visualize.py`.
Every result file is a `pickle` file, which contains an array of the following structure:

[

 [{input concepts as python set}, {output concepts as python set}, r^2 of estimation of output given input]
 
 ...

]

Every element in this array corresponds to description of some relation of input concepts towards
target concept. 

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
