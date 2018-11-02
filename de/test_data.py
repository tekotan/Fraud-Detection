import pandas as pd

tickets = pd.read_csv('data/closedtickets1.csv')
tkeys = tickets.keys()
tickets['comp']

kfi = open('keys.txt', 'w')
[kfi.write(tkeys[i] for i in range(len(tkeys)))] 
kfi.close()
