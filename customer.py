import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Customer_Segmentation_Dataset.csv")
    
    # Handle missing values
    df["Income"] = df["Income"].replace("", np.nan)
    df["Income"] = df["Income"].fillna(df["Income"].median())
    
    # Handle outliers in Income
    df["Income"] = df["Income"].clip(upper=df["Income"].quantile(0.99))
    
    # Encode categorical variables
    education_map = {"Basic": 1, "2n Cycle": 2, "Graduation": 3, "Master": 4, "PhD": 5}
    df["Education"] = df["Education"].map(education_map)
    
    # Simplify Marital_Status
    marital_map = {
        "Single": 0, "Divorced": 0, "Widow": 0, "Alone": 0, "Absurd": 0, "YOLO": 0,
        "Married": 1, "Together": 1
    }
    df["Marital_Status"] = df["Marital_Status"].map(marital_map)
    
    # Feature engineering
    current_year = 2025
    df["Age"] = current_year - df["Year_Birth"]
    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Total_Purchases"] = (
        df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    )
    df["Total_Dependents"] = df["Kidhome"] + df["Teenhome"]
    df["Campaign_Success"] = (
        df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]]
        .sum(axis=1) / 6
    )
    
    return df

# Perform clustering
def perform_clustering(df, n_clusters, features):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_data)
    
    return df, kmeans

# Compute elbow method
def compute_elbow_method(df, features, max_k=10):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    
    return inertias

# Streamlit app
st.title("Customer Segmentation Dashboard")
st.write("Analyze customer segments based on demographic and behavioral attributes.")

# Load data
df = load_data()

# Select features for clustering
features = [
    "Age", "Income", "Total_Spending", "Total_Purchases",
    "Total_Dependents", "Campaign_Success", "Education", "Marital_Status"
]

# User input for number of clusters
n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=4)

# Perform clustering
df_clustered, kmeans = perform_clustering(df, n_clusters, features)

# Elbow method plot
st.subheader("Elbow Method for Optimal Clusters")
inertias = compute_elbow_method(df, features)
fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(x=list(range(1, 11)), y=inertias, mode="lines+markers"))
fig_elbow.update_layout(
    title="Elbow Method",
    xaxis_title="Number of Clusters",
    yaxis_title="Inertia"
)
st.plotly_chart(fig_elbow)

# Scatter plot of clusters
st.subheader("Cluster Visualization")
fig_scatter = px.scatter(
    df_clustered,
    x="Income",
    y="Total_Spending",
    color="Cluster",
    title="Clusters by Income vs Total Spending",
    hover_data=["Age", "Total_Dependents"]
)
st.plotly_chart(fig_scatter)

# Segment characteristics
st.subheader("Segment Characteristics")
cluster_summary = df_clustered.groupby("Cluster")[features].mean().reset_index()
st.dataframe(cluster_summary)

# Bar chart of key features per cluster
st.subheader("Feature Distribution by Cluster")
fig_bar = go.Figure()
for feature in ["Age", "Income", "Total_Spending", "Campaign_Success"]:
    fig_bar.add_trace(
        go.Bar(
            x=cluster_summary["Cluster"],
            y=cluster_summary[feature],
            name=feature
        )
    )
fig_bar.update_layout(
    title="Average Feature Values per Cluster",
    xaxis_title="Cluster",
    yaxis_title="Value",
    barmode="group"
)
st.plotly_chart(fig_bar)

# Pie chart for segment sizes
st.subheader("Segment Size Distribution")
cluster_counts = df_clustered["Cluster"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]
fig_pie = px.pie(
    cluster_counts,
    names="Cluster",
    values="Count",
    title="Segment Size Distribution"
)
st.plotly_chart(fig_pie)

# Business insights
st.subheader("Business Insights")
for cluster in range(n_clusters):
    st.write(f"**Cluster {cluster}**:")
    cluster_data = cluster_summary[cluster_summary["Cluster"] == cluster]
    age = cluster_data["Age"].values[0]
    income = cluster_data["Income"].values[0]
    spending = cluster_data["Total_Spending"].values[0]
    campaign = cluster_data["Campaign_Success"].values[0]
    
    description = f"- **Average Age**: {age:.1f} years\n"
    description += f"- **Average Income**: ${income:.2f}\n"
    description += f"- **Average Total Spending**: ${spending:.2f}\n"
    description += f"- **Campaign Success Rate**: {campaign:.2%}\n"
    
    # Business relevance
    if income > df["Income"].mean() and spending > df["Total_Spending"].mean():
        description += "- **Business Implication**: High-value customers. Target with premium products and loyalty programs."
    elif campaign > 0.2:
        description += "- **Business Implication**: Responsive to campaigns. Focus on personalized promotions."
    else:
        description += "- **Business Implication**: Budget-conscious or less engaged. Offer discounts or entry-level products."
    
    st.markdown(description)