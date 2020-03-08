# Market basket analysis example
# (c) Nuno Antonio 2019

# To install mlxtend do "pip install mlxtend" (more info at http://rasbt.github.io/mlxtend/installation/)
# To install networkx do "pip install networkx" (more info at https://networkx.github.io)

# Import packages
import csv
import pandas as pd
import numpy as np
import datetime as dt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import networkx as nx

# Load and show dataset sample (Chain of Asian Restaurant sales January 2018)
import pandas as pd
dtypes = {'DocNumber':'category','ProductDesignation':'category'}
ds = pd.DataFrame(pd.read_csv("AsianRestaurant_Cyprus_2018_partial.txt",sep=";", dtype=dtypes))
ds.head()

# Describe dataset
ds.describe()
# 5096 transactions in 580 documents
# 203 different Products
# Max 32 lines per document
# Most sold item "Mineral water 1.5Lt" (632)

# Pivot the data - lines as documents and products as columns
pt = pd.pivot_table(ds, index='DocNumber', columns='ProductDesignation', aggfunc=lambda x: 1 if len(x)>0 else 0).fillna(0)
pt.head()

# Check in how many documents was SPRING ROLL sold
pt['SPRING ROLL'].sum()

# Apply the APRIORI algorithm
# Rules supported in at least 5% of the transactions (more info at http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
frequent_itemsets = apriori(pt, min_support=0.05, use_colnames=True)

# Generate the association rules - by confidence
rulesConfidence = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.50)
rulesConfidence.sort_values(by='confidence', ascending=False, inplace=True)
rulesConfidence.head(10)

# Generate the association rules - by lift
rulesLift = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
rulesLift.sort_values(by='lift', ascending=False, inplace=True)
rulesLift.head(10)

### Rules are in the form of "frozensets". Frozensets have functions to check if there are subsets, supersets, etc.
### More info at https://www.journaldev.com/22850/python-frozenset



##### EXPLORE FREQUENT_ITEMSETS #####

# Add a column with the length
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Length=2 and Support>=0.2
frequent_itemsets[(frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.2)]

# Spring Roll and Coke
frequent_itemsets[ frequent_itemsets['itemsets'] == {'SPRING ROLL', 'MINERAL WATER 1.5LT'}]

# Coke
frequent_itemsets[ frequent_itemsets['itemsets'] == {'COKE'}]

# High Confidence and high Lift
rulesConfidence[(rulesConfidence['confidence'] >= 0.9) & (rulesConfidence['lift'] >= 4)]

# High Confidence rules where "BEEF BBS" and "SPRING ROLL" are in the LHS
rulesConfidence[rulesConfidence['antecedents']=={'SPRING ROLL','BEEF BBS'}] # Because rules are a "frozenset" object, the order of items is not important

# High Confidence rules where "Sweet Sour Chick" is in the RHS
rulesConfidence[['SWEET SOUR CHICKEN' in elem for elem in rulesConfidence['consequents']]]

# Substitue products
rulesLift2 = association_rules(frequent_itemsets, metric="lift", min_threshold=0.0)
rulesLift2.sort_values(by='lift', ascending=True, inplace=True)
rulesLift2.head(10)




### Plot a basic network graph of the top 50 confidence rules
# Create a copy of the rules and transform the frozensets to strings
rulesToPlot = rulesConfidence.copy(deep=True)
rulesToPlot['LHS'] = [','.join(list(x)) for x in rulesToPlot['antecedents']]
rulesToPlot['RHS'] = [','.join(list(x)) for x in rulesToPlot['consequents']]
# Remove duplicate if reversed rules
rulesToPlot['sortedRow'] = [sorted([a,b]) for a,b in zip(rulesToPlot.LHS, rulesToPlot.RHS)]
rulesToPlot['sortedRow'] = rulesToPlot['sortedRow'].astype(str)
rulesToPlot.drop_duplicates(subset=['sortedRow'], inplace=True)
# Plot
rulesToPlot=rulesToPlot[:50]
fig = plt.figure(figsize=(20, 20)) 
G = nx.from_pandas_edgelist(rulesToPlot, 'LHS', 'RHS')
nx.draw(G, with_labels=True, node_size=30, node_color="red", pos=nx.spring_layout(G), seed=1234)
plt.axis('equal')
plt.show()
#fig.savefig('figure.svg')
