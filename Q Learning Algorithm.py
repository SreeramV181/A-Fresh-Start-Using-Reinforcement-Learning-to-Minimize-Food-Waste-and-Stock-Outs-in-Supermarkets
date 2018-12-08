import numpy as np
import random
import pickle
import pandas as pd
import joblib

#READ DATA

#Information on each store and item
stores = pd.read_csv("all/stores.csv") #store_nbr/city/state/type/cluster
print(stores.dtypes)

items = pd.read_csv("all/items.csv") #item_nbr/family/class/perishable
print(items.dtypes)

#Shipment Data, along with all possible item-store pairs (certain items only sold in certain stores)
shipments = pd.read_pickle("shipment_data.pkl")
store_item_pairs = pd.read_csv("store item pairs.csv")
policy_data = pd.concat([store_item_pairs, shipments], axis=1, join_axes=[shipments.index])

#Item numbers aren't linear (0...4100), so create mapping from their real ID to a linear ID
item_mapping = {}
for i in range(len(items)):
    item_mapping[items.iloc[i]['item_nbr']] = i
print(item_mapping[638977])


#PREPROCESS DATA

#Split data into training and testing
new_train = policy_data.iloc[0]['daily_results']
new_test = new_train.iloc[729:1094]
new_train = new_train.iloc[0:729]

#REINFORCEMENT LEARNING

#Implements Q-Learning Algorithm
Q = np.zeros((12, 22, 22))
for i in range(len(new_train)-2):

    #Access data for current date
    d = new_train.iloc[i]
    d1 = new_train.iloc[i+1]
    d2 = new_train.iloc[i+2]

    month = d['ds'].month
    inventory = int(d['inventory'])
    action = int(d['orders'])
    shipments = int(d['shipments'])
    sales = int(d['salesunits'])
    nextInventory = int(d1['inventory'])

    #Calculate reward (negative sum of out of stock and waste)
    oos = d['oos']
    waste = inventory + shipments - sales - nextInventory
    reward = -1.0 * (oos + waste)

    #Since our action or inventory space doesn't go beyond 21, function caps value at 21
    def ceil(x):
        if x > 20:
            return 21
        return int(x)

    #Ceilings Action and Inventory
    ceiledInventory = inventory
    if (inventory > 20):
        ceiledInventory = 21

    ceiledAction = action
    if (action > 20):
        ceiledAction = 21

    #Performs Q-Learning update
    maxQ = Q[month-1][ceil(inventory + shipments - sales)].max()
    #update Q based on month, ceiledInventory,
    Q[month - 1][ceiledInventory][ceiledAction] += .1 * (reward + .95 * maxQ -  Q[month - 1][ceiledInventory][ceiledAction])

#Tests Q-Learning model
theirTotal = 0
ourTotal = 0
for i in range(len(new_test)-2):
    d = new_train.iloc[i]
    d1 = new_train.iloc[i+1]

    month = d['ds'].month
    inventory = int(d['inventory'])

    action = Q[month-1][inventory].argmax()

    shipments = int(d['shipments'])
    sales = int(d['salesunits'])
    nextInventory = int(new_train.iloc[i+1]['inventory'])

    def ceil(x):
        if x > 20:
            return 21
        return int(x)

    ceiledInventory = inventory
    if (inventory > 20):
        ceiledInventory = 21

    #Calculate current option
    oos = d['oos']
    waste = inventory + shipments - sales - nextInventory
    theirTotal+=oos + waste

    #Calculate our option
    difference = action - shipments
    if (oos > 0 and difference > 0):
        our_oos = oos - difference
        if (our_oos < 0):
            our_oos = 0
    else:
        our_oos = oos + difference

    if (waste > 0 and difference < 0):
        our_waste = waste + difference
        if our_waste < 0:
            our_waste = 0
    else:
        our_waste = waste - difference

    ourTotal+=our_oos + our_waste
print('ours: ', ourTotal)
print('theirs: ', theirTotal)
