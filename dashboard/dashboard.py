import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Agile Bank Marketing Dashboard", layout="wide")

st.title("Agile Bank Marketing Dashboard")
st.write(
    "This dashboard supports stakeholder decision-making by analysing customer profiles "
    "and term deposit subscription patterns."
)

# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "bank.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Sidebar filters
st.sidebar.header("Interactive Filters")

job_filter = st.sidebar.selectbox(
    "Select Job Category",
    options=["All"] + sorted(df["job"].unique().tolist())
)

age_range = st.sidebar.slider(
    "Select Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (int(df["age"].min()), int(df["age"].max()))
)

filtered_df = df[
    (df["age"] >= age_range[0]) &
    (df["age"] <= age_range[1])
]

if job_filter != "All":
    filtered_df = filtered_df[filtered_df["job"] == job_filter]

# KPI summary
st.subheader("Dashboard Summary")

col1, col2, col3 = st.columns(3)

total_customers = len(filtered_df)
subscribed_customers = len(filtered_df[filtered_df["deposit"] == "yes"])
subscription_rate = (
    subscribed_customers / total_customers * 100
    if total_customers > 0 else 0
)

col1.metric("Total Customers", total_customers)
col2.metric("Subscribed Customers", subscribed_customers)
col3.metric("Subscription Rate", f"{subscription_rate:.2f}%")

# Visualization 1
st.subheader("Visualization 1: Deposit Subscription Distribution")

deposit_counts = filtered_df["deposit"].value_counts()

fig1, ax1 = plt.subplots()
ax1.bar(deposit_counts.index, deposit_counts.values)
ax1.set_xlabel("Deposit Subscription")
ax1.set_ylabel("Number of Customers")
ax1.set_title("Distribution of Deposit Subscription")
st.pyplot(fig1)

# Visualization 2
st.subheader("Visualization 2: Customer Age Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(filtered_df["age"], bins=20, edgecolor="black")
ax2.set_xlabel("Age")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Customer Age")
st.pyplot(fig2)

# Visualization 3
st.subheader("Visualization 3: Job Category by Deposit Subscription")

job_deposit = (
    filtered_df.groupby(["job", "deposit"])
    .size()
    .unstack(fill_value=0)
)

fig3, ax3 = plt.subplots(figsize=(10, 6))
job_deposit.plot(kind="barh", ax=ax3)
ax3.set_xlabel("Number of Customers")
ax3.set_ylabel("Job Category")
ax3.set_title("Job Category by Deposit Subscription")
st.pyplot(fig3)

# Filtered data table
st.subheader("Filtered Customer Records")
st.dataframe(filtered_df.head(20))
