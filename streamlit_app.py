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
            width: 300px;
            height: 100px; 
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
value = f"{total_rentals} rentals"
st.metric("Number of Rental", value=value)

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
    st.subheader("The Relationship Between Weather and The Bumber of Bicycle rentals")
    st.write(
        "We can use scatter plot to see the correlation pattern between the weather and count columns"
    )
    fig, ax = plt.subplots()
    sns.scatterplot(x=main_df["weather"], y=main_df["count"], ax=ax)
    sns.regplot(x=main_df["weather"], y=main_df["count"], ax=ax, scatter=False)
    ax.set_title("The relationship between weather and the number of bicycle rentals")
    st.pyplot(fig)
    st.write("The image above shows that the correlation between the two columns is very small.")

with tab2:
    st.subheader("Number and Average of Tenants by Working Day")
    st.write("We will use a bar plot to see the comparison between the average and the number of tenants on weekdays and non-weekdays.")
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
    ax[0].set_xlabel("Working day(1 = weekday)", fontsize=30)
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
    ax[1].set_xlabel("Working day(1 = weekday)", fontsize=30)
    ax[1].set_title("Number of tenants based on working days", loc="center", fontsize=36)
    ax[1].tick_params(axis="y", labelsize=28)
    ax[1].tick_params(axis="x", labelsize=28)

    def format_func(value, tick_number):
        return f"{value:.0f}"

    ax[1].yaxis.set_major_formatter(FuncFormatter(format_func))

    plt.suptitle("Number and average of tenants by working day", fontsize=50)

    st.pyplot(fig)

    st.write("Weekdays show larger numbers and averages on weekdays")

with tab3:
    st.subheader("Number of Rentals From Registered Renters During Weekends and Weekdays Base Pn Tenant Type")
    st.write("we can use plot points to see the daily borrowing pattern in hours based on tenant type (casual, registered)")
    fig1, ax1 = plt.subplots()
    sns.pointplot(data=main_df, x="hour", y="casual", hue="weekday", ax=ax1)
    ax1.set_title("Number of rentals from casual renters during weekends and weekdays")

    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.pointplot(data=main_df, x="hour", y="registered", hue="weekday", ax=ax2)
    ax2.set_title("Number of rentals from registered renters during weekends and weekdays")

    st.pyplot(fig2)

    st.write("The above pattern shows that casual renters rent more on weekends, while registered renters rent more on weekdays.")

# teknik analisis lanjutan

with tab4:
    st.subheader("Using Clustering Techniques to See Clusters From A Dataset")
    st.write("Use clustering techniques to see clusters of the dataset based on hour, season, weather, temperature, humidity, etc.")

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

    st.markdown(
        """
        From the results of grouping/clustering, it is divided into 4 clusters, namely:
- `Cluster 0` which can be defined that this cluster represents a weekday in the `afternoon in summer, with sunny weather and fairly warm temperatures`. The majority of users are `registered users` compared to `casual users`. This indicates that this cluster involves busy times during the after-work hours, with quite high bicycle use by regular users.
- `Cluster 1` which can be defined that this cluster describes `afternoon in spring, with cool temperatures and sunny weather`. Bicycle use by casual users is lower, and the majority of users are registered. This shows quite low activity compared to the previous cluster, perhaps because of the non-peak hours, or the cooler weather in spring.
- `Cluster 2` which can be defined that this cluster occurs in the `afternoon in summer with relatively warm temperatures`. This cluster may represent very busy times with a high total number of cyclists. This is likely to be the end of the work day or early evening when many people, both regular and occasional users, rent bikes. 
- `Cluster 3` which can be defined as representing `very early winter mornings with cold temperatures and high humidity`. Both casual and registered users are very low, indicating very little time spent on bicycles, perhaps due to the very early morning, cold weather and less than ideal conditions for cycling.
        """
    )

with tab5:
    st.subheader("Correlation Between Tables")
    st.write("Use heatmaps to display more interesting visualizations so that the correlation between tables can be seen clearly.")
    heatmap_df = main_df.drop(columns=["instant", "dateday", "year"], axis=1)
    correlation_values = heatmap_df.corr()
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(correlation_values, cbar=True, square=True, annot=True, annot_kws={"size": 8})
    st.pyplot(fig)
