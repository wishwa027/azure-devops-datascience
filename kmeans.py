

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Reading the data on which analysis needs to be done


def predict():
    retail = retail = pd.read_csv("./OnlineRetail.csv", encoding='unicode_escape')
    df_null = round(100 * (retail.isnull().sum()) / len(retail), 2)
    df_null
    # Droping rows having missing values
    retail = retail.dropna()
    # Changing the datatype of Customer Id as per Business understanding
    retail['CustomerID'] = retail['CustomerID'].astype(str)

    # New Attribute : Monetary

    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']
    rfm_m = retail.groupby('CustomerID')['Amount'].sum()
    rfm_m = rfm_m.reset_index()
    rfm_m.head()

    # New Attribute : Frequency

    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
    rfm_f = rfm_f.reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    rfm_f.head()

    # Merging the two dfs

    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm.head()

    # New Attribute : Recency

    # Convert to datetime to proper datatype

    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%m/%d/%Y %H:%M')

    # Compute the maximum date to know the last transaction date

    max_date = max(retail['InvoiceDate'])
    max_date

    # Compute the difference between max date and transaction date

    retail['Diff'] = max_date - retail['InvoiceDate']
    retail.head()

    # Compute last transaction date to get the recency of customers

    rfm_p = retail.groupby('CustomerID')['Diff'].min()
    rfm_p = rfm_p.reset_index()
    rfm_p.head()

    # Extract number of days only

    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm_p.head()

    rfm.head()

    rfm_p.head()

    rfm_p['CustomerID'] = rfm_p.CustomerID.astype(str)

    # Merge tha dataframes to get the final RFM dataframe

    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
    rfm.head()

    # Outlier Analysis of Amount Frequency and Recency

    attributes = ['Amount', 'Frequency', 'Recency']
    plt.rcParams['figure.figsize'] = [10, 8]
    sns.boxplot(data=rfm[attributes], orient="v", palette="Set2", whis=1.5, saturation=1, width=0.7)
    plt.title("Outliers Variable Distribution", fontsize=14, fontweight='bold')
    plt.ylabel("Range", fontweight='bold')
    plt.xlabel("Attributes", fontweight='bold')

    # Removing (statistical) outliers for Amount
    Q1 = rfm.Amount.quantile(0.05)
    Q3 = rfm.Amount.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Amount >= Q1 - 1.5 * IQR) & (rfm.Amount <= Q3 + 1.5 * IQR)]

    # Removing (statistical) outliers for Recency
    Q1 = rfm.Recency.quantile(0.05)
    Q3 = rfm.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Recency >= Q1 - 1.5 * IQR) & (rfm.Recency <= Q3 + 1.5 * IQR)]

    # Removing (statistical) outliers for Frequency
    Q1 = rfm.Frequency.quantile(0.05)
    Q3 = rfm.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Frequency >= Q1 - 1.5 * IQR) & (rfm.Frequency <= Q3 + 1.5 * IQR)]

    """### Rescaling the Attributes

    It is extremely important to rescale the variables so that they have a comparable scale.|
    There are two common ways of rescaling:

    1. Min-Max scaling
    2. Standardisation (mean-0, sigma-1)

    Here, we will use Standardisation Scaling.
    """

    # Rescaling the attributes

    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

    # Instantiate
    scaler = StandardScaler()

    # fit_transform
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled.shape

    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

    """<a id="4"></a> <br>
    ## Step 4 : Building the Model

    ### K-Means Clustering
    """

    # k-means with some arbitrary k

    kmeans = KMeans(n_clusters=4, max_iter=50)
    kmeans.fit(rfm_df_scaled)

    kmeans.labels_

    """### Finding the Optimal Number of Clusters

    #### Elbow Curve to get the right number of Clusters
    A fundamental step for any unsupervised algorithm is to determine the optimal number of clusters into which the data may be clustered. The Elbow Method is one of the most popular methods to determine this optimal value of k.
    """

    # Elbow-curve/SSD
    # inertia
    # Sum of squared distances of samples to their closest cluster center.

    ssd = []
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
        kmeans.fit(rfm_df_scaled)

        ssd.append(kmeans.inertia_)

    # Final model with k=3
    kmeans = KMeans(n_clusters=3, max_iter=50)
    kmeans.fit(rfm_df_scaled)

    kmeans.labels_

    # assign the label
    rfm['Cluster_Id'] = kmeans.labels_
    return rfm


"""### Inference:
K-Means Clustering with 3 Cluster Ids
- Customers with Cluster Id 1 are the customers with high amount of transactions as compared to other customers.
- Customers with Cluster Id 1 are frequent buyers.
- Customers with Cluster Id 2 are not recent buyers and hence least of importance from business point of view.
"""
