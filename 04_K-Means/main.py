import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read data
df = pd.read_excel("K_Means.xlsx")
print(df.head())

# Plot the data points
sns.regplot(x=df['X'], y=df['Y'], fit_reg=False)
# plt.savefig("seaborn_plt.png")
# plt.show()

# Create the instance of the model
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

# .fit() to train the model
model = kmeans.fit(df)

# .predict() to predict
predicted_values = kmeans.predict(df)

# Plot the clusters
# plt.scatter(df['X'], df['Y'])
plt.scatter(df['X'], df['Y'], c=predicted_values, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', alpha=0.9)
plt.savefig("kmeans_clusters.png")
plt.show()