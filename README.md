# Customer Segmentation Using RFM Analysis and K-Means Clustering

This project focuses on implementing customer segmentation for a UK-based retail chain using **Recency**, **Frequency**, and **Monetary (Revenue)** metrics. The aim is to optimize targeted marketing strategies and improve business outcomes through advanced analytics and machine learning.

## Objectives

1. **Decoding Customer Behavior**: Understand customer behavior using exploratory data analysis (EDA), advanced analytics, and machine learning.
2. **Targeted Marketing**: Develop targeted campaigns by analyzing purchasing patterns.
3. **Optimizing Retail Operations**: Use data-driven insights to enhance inventory management, pricing, and customer engagement strategies.

## Analytical Objectives

- **RFM Analysis**:
  - **Recency**: Measure how recently a customer made a purchase.
  - **Frequency**: Count how often a customer makes purchases.
  - **Monetary Value**: Evaluate how much a customer spends.
- **Clustering**:
  - Apply the **K-Means algorithm** to identify customer segments.
  - Use the **Elbow Method** to determine the optimal number of clusters.
  - Assign customers into high, mid, and low-value segments for better targeting.

## Dataset

The dataset comprises retail transactions from a UK chain store with 8 attributes and 541,909 records. It is publicly available at [Online Retail II Dataset](https://doi.org/10.24432/C5CG6D).

### Attributes:
1. **InvoiceNo**: Unique identifier for transactions.
2. **StockCode**: Product identifier.
3. **Description**: Product description.
4. **Quantity**: Quantity purchased.
5. **InvoiceDate**: Date and time of the transaction.
6. **UnitPrice**: Price per unit of the product.
7. **CustomerID**: Unique customer identifier.
8. **Country**: Country where the transaction occurred.

---

## Methodology

### Data Processing
1. **Loading**: Load the raw data into PySpark DataFrame.
2. **Cleaning**: Handle missing values and outliers.
3. **Feature Engineering**: Extract Recency, Frequency, and Revenue metrics.
4. **Scaling**: Standardize features to a common scale.
5. **Vectorization**: Assemble features into vectors for machine learning.

### Analytical Steps
1. **EDA**: Explore data distribution and relationships.
2. **RFM Metrics**:
   - Calculate Recency, Frequency, and Monetary values.
3. **Clustering**:
   - Use the **Elbow Method** to find the optimal number of clusters.
   - Apply **K-Means Clustering** on RFM metrics.
4. **Cluster Analysis**: Interpret the segments and derive actionable insights.

---

## Code Examples

### Elbow Method for Optimal Clusters

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

# Vectorize Revenue column
assembler = VectorAssembler(inputCols=["Revenue"], outputCol="features")
data = assembler.transform(tx_user)

# Elbow Method
sse = {}
for k in range(1, 10):
    kmeans = KMeans(featuresCol='features', k=k, maxIter=1000)
    model = kmeans.fit(data)
    sse[k] = model.summary.trainingCost

# Plot SSE vs. Number of Clusters
plt.plot(list(sse.keys()), list(sse.values()), marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.title("Elbow Method for Optimal k")
plt.show()


### K-Means Clustering
python
# Fit KMeans with optimal clusters (e.g., k=4)
kmeans = KMeans(featuresCol='features', k=4)
model = kmeans.fit(data)

# Assign clusters
clustered_data = model.transform(data)
clustered_data.show()


Results
Segmentation Insights
Low Value: Customers with low engagement and spending.
Mid Value: Moderately engaged and spending customers.
High Value: Loyal customers contributing the most revenue.

Key Metrics
Cluster	Avg Recency	Avg Frequency	Avg Revenue
Low: 200+ days	| <10 orders	| < $500
Mid:	100-200 days |	10-20 orders	| $500-$1000
High:	< 100 days |	>20 orders |	> $1000

Technologies Used
PySpark: Distributed data processing and machine learning.
Databricks: Collaborative data engineering and analytics.
AWS: Scalable infrastructure for data storage and processing.
Matplotlib: Visualization of clustering results.
