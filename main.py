'''
Plots the extracted relations from columns
'''

from relations_backend import Read_CSV_Columns, Extract_1_to_1_Relations

threshold = 0.5; # from -inf to 1; negative is bad, positive better than random

columns = Read_CSV_Columns("test_1.csv")
relations = Extract_1_to_1_Relations(columns, threshold);

# drawing procedures

