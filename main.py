import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter

from sklearn.cluster import KMeans
from sklearn.manifold import MDS

filename = "/Users/Shatalov/Downloads/European Jobs_data.csv"

df = pd.read_table(filename, sep=";")

data = df.iloc[:, 1:10].values

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 15), wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of cluster (k)')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4)


k = kmeans.fit_predict(data)

df['label'] = k

print(df)

cmd = MDS(n_components=2)
trans = cmd.fit_transform(data)


print(trans.shape)

plt.scatter(trans[k == 0, 0], trans[k == 0, 1], s=10, c='red', label='Cluster 1')
plt.scatter(trans[k == 1, 0], trans[k == 1, 1], s=10, c='blue', label='Cluster 2')
plt.scatter(trans[k == 2, 0], trans[k == 2, 1], s=10, c='green', label='Cluster 3')
plt.scatter(trans[k == 3, 0], trans[k == 3, 1], s=10, c='green', label='Cluster 4')
plt.show()

writer = pd.ExcelWriter('123.xlsx', engine='xlsxwriter')



