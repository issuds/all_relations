# Automated extraction of SEM structures

This software is a supplement to following publications: 

[ER18] Maass, W. & Shcherbatyi, I. Inductive Discovery By Machine Learning for Identification of Structural Models, The 37th International Conference on Conceptual Modeling (ER), 2018. 

[HICSS17] Maass, W. & Shcherbatyi, I. Data-Driven, Statistical Learning Method for Inductive Confirmation of Structural Models, Hawaii International Conference on System Sciences (HICSS), 2017. 

### Requirements

Python 3 is recommended to run the code. 
If something does not work on python 2 leat us know. 
You require the following packages to be installed on your system: 
`numpy`, `scipy`, `pandas` `scikit-learn`, `scikit-optimize`, `tqdm`. 

### Installation on Ubuntu

Open a terminal. In the terminal, navigate to the folder of the repository.
When in the folder, run following commands:

```bash
pip3 install -e .
```

This will install necessary python dependencies on your system. In order
to install necessary software for rendering of relations, use
```bash
bash install_dependencies_ubuntu.sh
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

Python script in the root folder of the repository can be used to reproduce results in the 
ER18 and HICSS17 publications. These do not require any configuration to run beyond 
installation of necessary dependencies. For example, to reproduce results for ER18
publication, run in terminal or execute in PyCharm

```bash
python3 er18_reproduce.py
```

For example datasets used, see `datasets` folder.

In order to use your own dataset, you first need to make sure that your dataset is in 
Comma Separated Value format (.csv). Secondly, you need to make sure that the columns
have a proper name format. The names of the columns should be given in the dataset, as 
they define the set of concepts, and what is considered a feature. In particular,
every column should be named as follows:

[concept]_[id],

where concept denotes the name of the concept,
and id denotes a particular indicator for a concept.
Underscores are not allowed in name or id, and can
break the program if present.
User features are specified using two special names
for concepts:
- respnum: numerical feature describing respondent
- respcat: categorical feature describing respondent

Example dataset is given below:
        
|respnum_Age | respcat_edu | Q1_a | Q1_b | Q2_a |
|------------|-------------|------|------|------|
|25          | College     | 1    | 2    | 5    |
|34          | University  | 2    | 3    | 4    |

Missing values are handled automatically and should not require
any manual preprocessing. Missing values are considered to be one
of ['', ' ', '?', 'NaN'].

## Structure of results

Results from the papers are stored in the `experimental_results` folder. 
You can load and print the results using the `allrelations.visualization`
functions (see docstrings). In particular, `render_relations` function
allows to convert JSON representation of results into visual graph form
and a csv table.