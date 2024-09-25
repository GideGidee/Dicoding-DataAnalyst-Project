import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter

sns.set(style="dark")


def create_avg_df(df):

    avg_rental_by_workingday = df.groupby(by="workingday").agg(
        {"count": ["sum", "mean"]}
    )
    avg_rental_by_workingday.columns = ["total_count", "avg_count"]
    avg_rental_by_workingday = avg_rental_by_workingday.reset_index()
    return avg_rental_by_workingday


def create_groupby_day_df(df):
    group_by_day_df = (
        df.groupby(by="dateday")
        .agg({"weekday": "nunique", "workingday": "nunique", "count": "sum"})
        .reset_index()
    )
    return group_by_day_df


all_df = pd.read_csv("data/all_data.csv")

all_df = all_df.rename(
    columns={
        "yr": "year",
        "mnth": "month",
        "hum": "humidity",
        "weathersit": "weather",
        "cnt": "count",
        "hr": "hour",
        "dteday": "dateday",
    }
)

all_df["dateday"] = pd.to_datetime(all_df["dateday"])

min_date = all_df["dateday"].min()
max_date = all_df["dateday"].max()

with st.sidebar:
    st.markdown(
        """
        <style>
        .fixed-image {
            width: 300px; /* Ubah ukuran sesuai keinginan */
            height: 100px; /* Ubah ukuran sesuai keinginan */
        }
        </style>
        <img src="https://storage.googleapis.com/kaggle-datasets-images/130897/312329/20c79bcd928e6d481fca7d5dc9fa3ca4/dataset-cover.jpg?t=2019-05-24-07-06-55" class="fixed-image">
        """,
        unsafe_allow_html=True,
    )

    start_date, end_date = st.date_input(
        label="Rentang waktu",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date],
    )

main_df = all_df[
    (all_df["dateday"] >= pd.to_datetime(start_date))
    & (all_df["dateday"] <= pd.to_datetime(end_date))
]

avg_rental_by_workingday = create_avg_df(main_df)
group_by_day_df = create_groupby_day_df(main_df)

st.title("Analyzing Bike Sharing Data Dashboard :sparkles:")

st.subheader("Daily Rental")

total_rentals = main_df["count"].sum()
st.metric("Total rental", value=total_rentals)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    group_by_day_df["dateday"],
    group_by_day_df["count"],
    marker="o",
    linewidth=2,
    color="#90CAF9",
)
ax.tick_params(axis="y", labelsize=20)
ax.tick_params(axis="x", labelsize=15)

st.pyplot(fig)

st.header("Result of Analizing Bike Sharing Data")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Relation Weather and Count", "Avg & Sum", "Pattern by Weekday", "Clustering", "Correlations"]
)

with tab1:
    fig, ax = plt.subplots()
    sns.scatterplot(x=main_df["weather"], y=main_df["count"], ax=ax)
    sns.regplot(x=main_df["weather"], y=main_df["count"], ax=ax, scatter=False)
    ax.set_title("The relationship between weather and the number of bicycle rentals")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(24, 30))
    colors = ["#72BCD4", "#D3D3D3"]

    sns.barplot(
        x="workingday",
        y="avg_count",
        data=avg_rental_by_workingday,
        palette=colors,
        ax=ax[0],
    )
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("Working day(1 = Workday)", fontsize=30)
    ax[0].set_title("Average tenants by working day", loc="center", fontsize=36)
    ax[0].tick_params(axis="y", labelsize=28)
    ax[0].tick_params(axis="x", labelsize=28)

    sns.barplot(
        x="workingday",
        y="total_count",
        data=avg_rental_by_workingday,
        palette=colors,
        ax=ax[1],
    )
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("Working day(1 = Workday)", fontsize=30)
    ax[1].set_title(
        "Number of tenants based on working days", loc="center", fontsize=36
    )
    ax[1].tick_params(axis="y", labelsize=28)
    ax[1].tick_params(axis="x", labelsize=28)

    def format_func(value, tick_number):
        return f"{value:.0f}"

    ax[1].yaxis.set_major_formatter(FuncFormatter(format_func))

    plt.suptitle("Number and average of tenants by working day", fontsize=50)

    st.pyplot(fig)

with tab3:
    fig1, ax1 = plt.subplots()
    sns.pointplot(data=main_df, x="hour", y="casual", hue="weekday", ax=ax1)
    ax1.set_title("Number of rentals from casual renters during weekends and weekdays")

    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.pointplot(data=main_df, x="hour", y="registered", hue="weekday", ax=ax2)
    ax2.set_title(
        "Number of rentals from registered renters during weekends and weekdays"
    )

    st.pyplot(fig2)

# teknik analisis lanjutan

with tab4:
    from sklearn.preprocessing import StandardScaler

    # cari jumlah cluster optimal
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(
        main_df[
            [
                "casual",
                "registered",
                "count",
                "temp",
                "humidity",
                "windspeed",
                "hour",
                "workingday",
                "holiday",
            ]
        ]
    )

    from sklearn.cluster import KMeans

    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        sse.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 11), sse)
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("SSE")
    ax1.set_title("Finding the optimal number of clusters with the Elbow Method")
    st.pyplot(fig1)

    # tampilkan scatter dari cluster
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(data_scaled)
    main_df["cluster"] = kmeans.labels_

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    fig2, ax2 = plt.subplots()
    ax2.scatter(data_pca[:, 0], data_pca[:, 1], c=main_df["cluster"], cmap="viridis")
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.set_title("Clustering Visualization")
    st.pyplot(fig2)

    # menampilkan tabel clustering
    st.subheader("Cluster Table")
    cluster_df = main_df.groupby(by="cluster").mean()

    # Pilih kolom yang diinginkan untuk ditampilkan
    selected_columns = [
        "hour",
        "season",
        "weather",
        "temp",
        "humidity",
        "casual",
        "registered",
    ]

    # Filter DataFrame hanya dengan kolom yang dipilih
    filtered_cluster_df = cluster_df[selected_columns]

    # Menampilkan tabel di Streamlit
    st.dataframe(data=filtered_cluster_df, width=700)

with tab5:
    heatmap_df = main_df.drop(columns=["instant", "dteday", "year"], axis=1)
    correlation_values = heatmap_df.corr()
    fig, ax = plt.subplots(figsize=(15,15))
    ax.heatmap(
        correlation_values, cbar=True, square=True, annot=True, annot_kws={"size": 8}
    )
    st.pyplot(fig)