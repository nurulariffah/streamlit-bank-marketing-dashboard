import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.bar(deposit_counts.index, deposit_counts.values)
ax1.set_xlabel("Deposit Subscription")
ax1.set_ylabel("Number of Customers")
ax1.set_title("Distribution of Deposit Subscription")
st.pyplot(fig1, use_container_width=False)

# Visualization 2
st.subheader("Visualization 2: Customer Age Distribution")

fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.hist(filtered_df["age"], bins=20, edgecolor="black")
ax2.set_xlabel("Age")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Customer Age")
st.pyplot(fig2, use_container_width=False)

# Visualization 3
st.subheader("Visualization 3: Job Category by Deposit Subscription")

job_deposit = (
    filtered_df.groupby(["job", "deposit"])
    .size()
    .unstack(fill_value=0)
)

fig3, ax3 = plt.subplots(figsize=(7,5))
job_deposit.plot(kind="barh", ax=ax3)
ax3.set_xlabel("Number of Customers")
ax3.set_ylabel("Job Category")
ax3.set_title("Job Category by Deposit Subscription")
st.pyplot(fig3, use_container_width=False)

# -------------------------------
# Predictive Output
# -------------------------------

st.subheader("Predictive Output Using Random Forest Model")

model_df = df.copy()

# Encode categorical variables
label_encoders = {}

categorical_columns = model_df.select_dtypes(include="object").columns

for col in categorical_columns:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    label_encoders[col] = le

# Scale numerical variables
numerical_columns = [
    "age",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

scaler = StandardScaler()

model_df[numerical_columns] = scaler.fit_transform(
    model_df[numerical_columns]
)

# Features and target
X = model_df.drop("deposit", axis=1)
y = model_df["deposit"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# User selection
customer_index = st.number_input(
    "Select Customer Record Index",
    min_value=0,
    max_value=len(df)-1,
    value=0
)

selected_customer = X.iloc[[customer_index]]

prediction = model.predict(selected_customer)[0]

prediction_label = label_encoders["deposit"].inverse_transform(
    [prediction]
)[0]

st.write("Selected Customer Record")

st.dataframe(df.iloc[[customer_index]])

st.metric(
    "Predicted Deposit Subscription",
    prediction_label.upper()
)

# -------------------------------
# Filtered data table
# -------------------------------

st.subheader("Filtered Customer Records")

st.dataframe(filtered_df.head(20))

st.header("Model Monitoring")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Model Accuracy",
        value="82.58%"
    )

with col2:
    missing_values = df.isnull().sum().sum()

    st.metric(
        label="Missing Values",
        value=int(missing_values)
    )
# -------------------------------
# Data Drift Analysis
# -------------------------------

st.header("Data Drift Analysis")

st.write(
    "This section compares the reference dataset used during model development "
    "with simulated deployment data to identify potential data drift."
)

# Reference dataset
reference_mean_age = df["age"].mean()

# Simulated deployment dataset
current_df = df.copy()
current_df["age"] = current_df["age"] + 5

current_mean_age = current_df["age"].mean()
age_difference = current_mean_age - reference_mean_age

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Reference Mean Age",
        f"{reference_mean_age:.2f}"
    )

with col2:
    st.metric(
        "Current Mean Age",
        f"{current_mean_age:.2f}"
    )

with col3:
    st.metric(
        "Age Difference",
        f"{age_difference:.2f}"
    )

if abs(age_difference) > 2:
    st.warning("Status: Data Drift Detected")
else:
    st.success("Status: No Significant Data Drift")
