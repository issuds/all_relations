# in order to reduce the communication overhead
export MKL_NUM_THREADS=1

#python3 -W ignore main.py lasso
#python3 -W ignore main.py knn
#python3 -W ignore main.py gbrt
#python3 -W ignore main.py ann
python3 -W ignore main.py tree
python3 -W ignore main.py lasso --prefix
#python3 -W ignore main.py ann --prefix
python3 -W ignore main.py knn --prefix
python3 -W ignore main.py gbrt --prefix
python3 -W ignore main.py tree --prefix