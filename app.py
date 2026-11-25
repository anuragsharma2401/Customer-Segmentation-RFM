import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Load trained models
kmeans = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("outputs/customer_segmentation_results.csv")

    # Handle customer ID stored as CSV index
    if "Customer ID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Customer ID"})

    return df


rfm = load_data()

# --------------------------------------------------
# SEGMENT INFORMATION
# --------------------------------------------------

segment_strategies = {
    "Champions": {
        "description": "Recent, frequent and high-spending customers.",
        "strategy": "Reward loyalty, provide exclusive offers and encourage referrals.",
        "priority": "Retention"
    },

    "At-Risk Valuable": {
        "description": "Previously valuable customers who have not purchased recently.",
        "strategy": "Launch personalized win-back campaigns and collect feedback.",
        "priority": "Reactivation"
    },

    "Recent Low-Frequency": {
        "description": "Recently active customers with relatively low purchase frequency.",
        "strategy": "Encourage repeat purchases through recommendations and targeted offers.",
        "priority": "Engagement"
    },

    "Hibernating": {
        "description": "Customers with high recency and relatively low historical spending.",
        "strategy": "Use low-cost reactivation campaigns and monitor responses.",
        "priority": "Reactivation"
    }
}

# --------------------------------------------------
# CALCULATE BUSINESS METRICS
# --------------------------------------------------

total_customers = len(rfm)

total_revenue = rfm["Monetary"].sum()

avg_spending = rfm["Monetary"].mean()

total_orders = rfm["Frequency"].sum()

avg_recency = rfm["Recency"].mean()

segment_counts = (
    rfm["Segment"]
    .value_counts()
    .rename_axis("Segment")
    .reset_index(name="Customers")
)

segment_counts["Percentage"] = (
    segment_counts["Customers"] / total_customers * 100
)

segment_revenue = (
    rfm.groupby("Segment")["Monetary"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

segment_revenue["Revenue Percentage"] = (
    segment_revenue["Monetary"] / total_revenue * 100
)

segment_profile = rfm.groupby("Segment").agg(
    Customers=("Recency", "count"),
    Recency=("Recency", "median"),
    Frequency=("Frequency", "median"),
    Monetary=("Monetary", "median")
).reset_index()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📊 Customer Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Segment Analysis",
        "Customer Explorer",
        "Business Recommendations",
        "Customer Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Customer segmentation using "
    "RFM Analysis and K-Means Clustering."
)

# ==================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==================================================

if page == "Executive Overview":

    st.title("📊 Customer Intelligence Dashboard")

    st.markdown(
        "Understand customer behaviour, purchasing patterns "
        "and revenue contribution through machine learning."
    )

    st.divider()

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Total Revenue",
        f"£{total_revenue:,.2f}"
    )

    col3.metric(
        "Average Spending",
        f"£{avg_spending:,.2f}"
    )

    col4.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    st.divider()

    # CUSTOMER DISTRIBUTION AND REVENUE

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Distribution")

        fig = px.pie(
            segment_counts,
            names="Segment",
            values="Customers",
            hole=0.5
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Revenue Contribution")

        fig = px.bar(
            segment_revenue,
            x="Segment",
            y="Monetary",
            color="Segment",
            text_auto=".2s"
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Customer Segment",
            yaxis_title="Revenue (£)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # AUTOMATED BUSINESS INSIGHTS

    st.subheader("💡 Key Business Insights")

    highest_revenue_segment = segment_revenue.iloc[0]

    largest_segment = segment_counts.sort_values(
        "Customers",
        ascending=False
    ).iloc[0]

    champions = segment_profile[
        segment_profile["Segment"] == "Champions"
    ]

    if not champions.empty:

        champion_data = champions.iloc[0]

        st.success(
            f"**Top revenue segment:** "
            f"{highest_revenue_segment['Segment']} contributes "
            f"{highest_revenue_segment['Revenue Percentage']:.1f}% "
            f"of total historical revenue."
        )

        st.info(
            f"**Largest customer segment:** "
            f"{largest_segment['Segment']} contains "
            f"{largest_segment['Customers']:,} customers "
            f"({largest_segment['Percentage']:.1f}% of the customer base)."
        )

        st.warning(
            f"**Champions:** Median purchase frequency is "
            f"{champion_data['Frequency']:.0f} orders, "
            f"with median spending of "
            f"£{champion_data['Monetary']:,.2f}."
        )

    st.caption(
        "Insights are calculated from historical transaction data. "
        "They are descriptive, not predictions of future behaviour."
    )


# ==================================================
# PAGE 2: SEGMENT ANALYSIS
# ==================================================

elif page == "Segment Analysis":

    st.title("🔍 Customer Segment Analysis")

    st.write(
        "Compare the purchasing behaviour of different "
        "customer groups."
    )

    st.divider()

    st.subheader("RFM Segment Profiles")

    st.dataframe(
        segment_profile.style.format({
            "Customers": "{:,.0f}",
            "Recency": "{:.0f} days",
            "Frequency": "{:.0f}",
            "Monetary": "£{:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # RFM COMPARISON

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Median Frequency")

        fig = px.bar(
            segment_profile,
            x="Segment",
            y="Frequency",
            color="Segment",
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Median Monetary Value")

        fig = px.bar(
            segment_profile,
            x="Segment",
            y="Monetary",
            color="Segment",
            text_auto=".2s"
        )

        fig.update_layout(
            yaxis_title="Monetary (£)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Customer Recency vs Monetary Value")

    fig = px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        color="Segment",
        hover_data=["Customer ID", "Frequency"],
        log_y=True,
        opacity=0.6
    )

    fig.update_layout(
        xaxis_title="Recency (Days)",
        yaxis_title="Monetary Value (£, Log Scale)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Lower recency indicates more recent purchases. "
        "The monetary axis uses a logarithmic scale to handle "
        "the highly skewed spending distribution."
    )


# ==================================================
# PAGE 3: CUSTOMER EXPLORER
# ==================================================

elif page == "Customer Explorer":

    st.title("👥 Customer Explorer")

    st.write(
        "Search and investigate individual customer behaviour."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        selected_segment = st.selectbox(
            "Select Customer Segment",
            ["All"] + sorted(rfm["Segment"].unique().tolist())
        )

    with col2:

        customer_search = st.text_input(
            "Search Customer ID"
        )

    filtered_data = rfm.copy()

    if selected_segment != "All":

        filtered_data = filtered_data[
            filtered_data["Segment"] == selected_segment
        ]

    if customer_search:

        filtered_data = filtered_data[
            filtered_data["Customer ID"].astype(str).str.contains(
                customer_search,
                na=False
            )
        ]

    st.write(
        f"Showing {len(filtered_data):,} customers"
    )

    st.dataframe(
        filtered_data[
            [
                "Customer ID",
                "Recency",
                "Frequency",
                "Monetary",
                "Segment"
            ]
        ].sort_values("Monetary", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    csv = filtered_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Customer Report",
        data=csv,
        file_name="customer_segmentation_report.csv",
        mime="text/csv"
    )
    
    st.divider()

    if customer_search and len(filtered_data) == 1:

        customer = filtered_data.iloc[0]

        st.subheader("Customer Details")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Recency",
            f"{customer['Recency']} days"
        )

        col2.metric(
            "Frequency",
            f"{customer['Frequency']} orders"
        )

        col3.metric(
            "Monetary",
            f"£{customer['Monetary']:,.2f}"
        )

        st.success(
            f"Customer Segment: {customer['Segment']}"
        )


# ==================================================
# PAGE 4: BUSINESS RECOMMENDATIONS
# ==================================================

elif page == "Business Recommendations":

    st.title("💼 Business Recommendations")

    st.write(
        "Convert machine learning results into "
        "actionable marketing strategies."
    )

    st.divider()

    for segment, details in segment_strategies.items():

        segment_data = segment_profile[
            segment_profile["Segment"] == segment
        ]

        if segment_data.empty:
            continue

        data = segment_data.iloc[0]

        with st.expander(
            f"📌 {segment}",
            expanded=True
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Customers",
                    f"{data['Customers']:,}"
                )

                st.metric(
                    "Median Spending",
                    f"£{data['Monetary']:,.2f}"
                )

            with col2:

                st.metric(
                    "Median Frequency",
                    f"{data['Frequency']:.0f} orders"
                )

                st.metric(
                    "Median Recency",
                    f"{data['Recency']:.0f} days"
                )

            st.write(
                f"**Customer Behaviour:** {details['description']}"
            )

            st.write(
                f"**Recommended Strategy:** {details['strategy']}"
            )

            st.write(
                f"**Business Priority:** {details['priority']}"
            )

    st.divider()

    st.subheader("Model Information")

    st.write("""
    - Algorithm: K-Means Clustering
    - Feature Engineering: RFM Analysis
    - Number of Clusters: 4
    - Feature Transformation: Logarithmic Transformation
    - Feature Scaling: StandardScaler
    - Silhouette Score: 0.3354
    """)

    st.caption(
        "The four-segment solution was selected for business "
        "interpretability, although K=2 achieved a higher "
        "silhouette score of 0.4135."
    )

# ==================================================
# PAGE 5: CUSTOMER PREDICTION
# ==================================================

elif page == "Customer Prediction":

    st.title("🤖 Customer Segment Prediction")

    st.write(
        "Predict the customer segment using "
        "Recency, Frequency and Monetary values."
    )

    st.divider()

    st.subheader("Enter Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Recency (Days)",
            min_value=1,
            max_value=739,
            value=30
        )

    with col2:
        frequency = st.number_input(
            "Frequency (Orders)",
            min_value=1,
            max_value=12425,
            value=10
        )

    with col3:
        monetary = st.number_input(
            "Monetary Value (£)",
            min_value=2.95,
            value=500.0
        )

    if st.button("Predict Customer Segment", type="primary"):

        import numpy as np

        # Create RFM input
        customer_data = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        # Apply same transformation as training
        customer_log = np.log1p(customer_data)

        # Apply saved scaler
        customer_scaled = scaler.transform(customer_log)

        # Predict cluster
        predicted_cluster = kmeans.predict(customer_scaled)[0]

        # Convert cluster into business segment
        segment_names = {
            0: "At-Risk Valuable",
            1: "Champions",
            2: "Recent Low-Frequency",
            3: "Hibernating"
        }

        predicted_segment = segment_names[predicted_cluster]

        st.success(
            f"Predicted Customer Segment: {predicted_segment}"
        )
        st.divider()
        st.subheader("Customer RFM Profile")

        col1, col2, col3 = st.columns(3)

        col1.metric("Recency", f"{recency} days")
        col2.metric("Frequency", f"{frequency} orders")
        col3.metric("Monetary", f"£{monetary:,.2f}")

        st.subheader("Segment Insights")

        st.write(segment_strategies[predicted_segment]["description"])

        st.info(
            f"Recommended Strategy: "
            f"{segment_strategies[predicted_segment]['strategy']}"
        )

        st.warning(
            f"Business Priority: "
            f"{segment_strategies[predicted_segment]['priority']}"
        )

        st.info(
            segment_strategies[predicted_segment]["strategy"]
        )