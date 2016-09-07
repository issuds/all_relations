# all_relations

The most certain way to install everything is to use Ubuntu virtual machine. You require virtualbox to be installed:

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

Create bitbucket account and request rights for all_relations repository. Clone the repository in some folder. Open Pycharm IDE, and open the folder as new project.
To run the code, right click on "main.py" in the project manager (left panel in pycharm app, double click on "all_relations" in the left upper corner if you cannot find it) and click "run main".
wiki.py is the main script that can be used to reproduce our results. All the data we use is stored in "datasets" folder in the pycharm project.
By default, the results of computation are cached in the file 'wiki.bin'. If you make changes to the code, rename this file so that results are recomputed.
For details on the data format refer to "dataset_parser.py".

The most straightforward way to share data with virtual machine is through cloud data storage (e.g. dropbox, google drive).
