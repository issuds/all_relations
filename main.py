'''
Plots the extracted relations from columns
'''

import matplotlib.pyplot as plt
import networkx as nx
from relations_backend import Read_CSV_Columns, Extract_1_to_1_Relations

threshold = 0.5; # from -inf to 1; negative is bad, positive better than random

columns = Read_CSV_Columns("test_1.csv")
relations = Extract_1_to_1_Relations(columns, threshold);

# drawing procedures

G=nx.DiGraph()

for relation in relations:
    G.add_edge(relation[0],relation[1],weight=relation[2])


elarge=[(u,v) for (u,v,d) in G.edges(data=True) ]

pos=nx.spring_layout(G) # positions for all nodes
nx.draw_networkx_nodes(G,pos,node_size=700)
nx.draw_networkx_edges(G,pos,edgelist=elarge,
                    width=6)

# labels
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
nx.draw_networkx_edge_labels(G,pos,font_size=10,font_family='sans-serif')

plt.axis('off')
plt.show() # display