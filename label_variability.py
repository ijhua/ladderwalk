import pandas as pd
import numpy as np
import math

df = pd.read_hdf("/Users/mathuser/Documents/CollectedData_Isabelle.h5")

df = df["Isabelle"]
df = df.reset_index()
df["folder"]=df['index'].apply(lambda x: pd.Series(str(x).split("/")[1]))
df = df.drop(['index'],axis=1)
folders = list(set(df['folder'].tolist()))

def dist(coor1,coor2):
    x1 = float(coor1[0])
    y1 = float(coor1[1])
    x2 = float(coor2[0])
    y2 = float(coor2[1])
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

rung_list = []
for i in range(1,63):
    rung_list.append("rung_"+str(i))

lst = []
for f in folders:
    tdf = df.loc[df['folder']==f]
    tdf = tdf.reset_index()
    for r in rung_list:
        rdf = tdf[r]
        rdf = rdf.dropna()
        rdf = rdf.reset_index()
        coords = []
        for i in range(len(rdf.index)):
            coords.append((rdf['x'][i],rdf['y'][i]))
        if len(coords)==3:
            dist1 = dist(coords[0],coords[1])
            dist2 = dist(coords[1],coords[2])
            dist3 = dist(coords[0],coords[2])
        if len(coords)==2:
            dist1 = dist(coords[0],coords[1])
            dist2 = np.nan
            dist3 = np.nan
        if len(coords)==1:
            dist1 = np.nan
            dist2 = np.nan
            dist3 = np.nan
        lst.append([f,r,dist1,dist2,dist3])

df2 = pd.DataFrame(lst,columns = ["folder","rung","distance_1","distance_2","distance_3"])

dists = []
dists.append(df2.distance_1.dropna().values)
dists.append(df2.distance_2.dropna().values)
dists.append(df2.distance_3.dropna().values)

print("Mean: "+str(dists[0].mean())+" px")

print("Standard Deviation: "+str(dists[0].std())+" px")
