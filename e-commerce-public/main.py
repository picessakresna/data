import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import streamlit as st
import urllib
from func import DataAnalyzer, BrazilMapPlotter
from babel.numbers import format_currency
sns.set(style='dark')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Dataset
datetime_cols = ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", "order_purchase_timestamp", "shipping_limit_date"]
all_df = pd.read_csv("./dataset/all_data.csv")
all_df.sort_values(by="order_approved_at", inplace=True)
all_df.reset_index(inplace=True)

# Geolocation Dataset
geolocation = pd.read_csv('./dataset/geolocation.csv')
data = geolocation.drop_duplicates(subset='customer_unique_id')

for col in datetime_cols:
    all_df[col] = pd.to_datetime(all_df[col])

min_date = all_df["order_approved_at"].min()
max_date = all_df["order_approved_at"].max()

# Sidebar
with st.sidebar:
    # Title
    st.title("Picessa Kresna")

    # Logo Image
    st.image("gcl.png")

    # Date Range
    start_date, end_date = st.date_input(
        label="Pilih Tanggal",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Main
main_df = all_df[(all_df["order_approved_at"] >= str(start_date)) & 
                 (all_df["order_approved_at"] <= str(end_date))]

function = DataAnalyzer(main_df)
map_plot = BrazilMapPlotter(data, plt, mpimg, urllib, st)

daily_orders_df = function.create_daily_orders_df()
sum_spend_df = function.create_sum_spend_df()
sum_order_items_df = function.create_sum_order_items_df()
review_score, common_score = function.review_score_df()
state, most_common_state = function.create_bystate_df()
order_status, common_status = function.create_order_status()

# Title
st.title('Dashboard E-Commerce Public')

# Daily Orders
st.subheader("Order Harian")

col1, col2 = st.columns(2)

with col1:
    total_order = daily_orders_df["order_count"].sum()
    st.markdown(f"Total Order: **{total_order}**")

with col2:
    total_revenue = format_currency(daily_orders_df["revenue"].sum(), "IDR", locale="id_ID")
    st.markdown(f"Total Keuntungan: **{total_revenue}**")

max_order_date = daily_orders_df[daily_orders_df["order_count"] == daily_orders_df["order_count"].max()]["order_approved_at"].dt.date.iloc[0]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    daily_orders_df["order_approved_at"],
    daily_orders_df["order_count"],
    marker="o",
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", labelsize=15)
st.pyplot(fig)

with st.expander("Lihat Penjelasan"):
    st.write(f"Sesuai dengan grafik yang sudah dibuat, pada order harian, diketahui bahwa total order yaitu {total_order} order dan memiliki total keuntungan yaitu {total_revenue} dengan tanggal dengan order tertinggi yaitu {max_order_date}")


# Customer Spend Money
st.subheader("Uang Yang Dibelanjakan Konsumen")
col1, col2 = st.columns(2)

with col1:
    total_spend = format_currency(sum_spend_df["total_spend"].sum(), "IDR", locale="id_ID")
    st.markdown(f"Total Uang yang Dibelanjakan Konsumen: **{total_spend}**")

with col2:
    avg_spend = format_currency(sum_spend_df["total_spend"].mean(), "IDR", locale="id_ID")
    st.markdown(f"Rata-Rata Uang yang Dibelanjakan Konsumen: **{avg_spend}**")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    sum_spend_df["order_approved_at"],
    sum_spend_df["total_spend"],
    marker="o",
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", labelsize=15)
st.pyplot(fig)

with st.expander("Lihat Penjelasan"):
    st.write(f"Sesuai dengan grafik yang sudah dibuat, pada uang yang dibelanjakan konsumen, diketahui bahwa total uang yang dibelanjakan konsumen yaitu {total_spend} dan rata-rata uang yang dibelanjakan konsumen yaitu {avg_spend} dengan tanggal dengan order tertinggi yaitu {max_order_date}")

# Order Items
st.subheader("Barang yang di Order")
col1, col2 = st.columns(2)

with col1:
    total_items = sum_order_items_df["product_count"].sum()
    st.markdown(f"Total Barang: **{total_items}**")

with col2:
    avg_items = sum_order_items_df["product_count"].mean()
    st.markdown(f"Rata-Rata Barang: **{avg_items}**")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(45, 25))

colors = ["#068DA9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

# First barplot
sns.barplot(
    x="product_count", 
    y="product_category_name_english", 
    data=sum_order_items_df.head(5),
    legend=False,
    hue="product_category_name_english", 
    palette=colors, 
    ax=ax[0]
)

ax[0].set_ylabel(None)
ax[0].set_xlabel("Jumlah Penjualan", fontsize=30)
ax[0].set_title("Produk paling banyak terjual", loc="center", fontsize=50)
ax[0].tick_params(axis ='y', labelsize=35)
ax[0].tick_params(axis ='x', labelsize=30)

# Second barplot
sns.barplot(
    x="product_count", 
    y="product_category_name_english", 
    data=sum_order_items_df.sort_values(by="product_count", ascending=True).head(5), 
    palette=colors,
    legend=False,
    hue="product_category_name_english",  
    ax=ax[1]
)
ax[1].set_ylabel(None)
ax[1].set_xlabel("Jumlah Penjualan", fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Produk paling sedikit terjual", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

st.pyplot(fig)

most_sold_item = sum_order_items_df.loc[sum_order_items_df["product_count"].idxmax(), "product_category_name_english"]
least_sold_item = sum_order_items_df.loc[sum_order_items_df["product_count"].idxmin(), "product_category_name_english"]

with st.expander("Lihat Penjelasan"):
    st.write(f"Sesuai dengan grafik yang sudah dibuat, pada barang yang diorder, diketahui barang yang paling banyak terjual yaitu **{most_sold_item}** dan barang yang paling sedikit terjual yaitu **{least_sold_item}**")

# Review Score
st.subheader("Skor Review")
col1,col2 = st.columns(2)

with col1:
    avg_review_score = review_score.mean()
    st.markdown(f"Rata-rata Skor Review: **{avg_review_score}**")

with col2:
    most_common_review_score = review_score.value_counts().index[0]
    st.markdown(f"Skor Review Terbanyak: **{most_common_review_score}**")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=review_score.index, 
            y=review_score.values,
            legend=False,  
            order=review_score.index,
            palette=["#068DA9" if score == common_score else "#D3D3D3" for score in review_score.index]
            )

plt.title("Penilaian dari konsumen untuk pelayanan", fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Jumlah")
plt.xticks(fontsize=12)
st.pyplot(fig)

# Customer Demographic
st.subheader("Demografi Konsumen Per Kota")
most_common_state = state.customer_state.value_counts().index[0]
st.markdown(f"Kota Paling Banyak: **{most_common_state}**")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=state.customer_state.value_counts().index,
            y=state.customer_count.values,
            hue=state.customer_state.value_counts().index,
            data=state,
            palette=["#068DA9" if score == most_common_state else "#D3D3D3" for score in state.customer_state.value_counts().index]
            )

plt.title("Jumlah Konsumen dari Kota", fontsize=15)
plt.xlabel("Kota")
plt.ylabel("Jumlah Konsumen")
plt.xticks(fontsize=12)
st.pyplot(fig)

with st.expander("Lihat Penjelasan"):
    st.write(f"Sesuai dengan grafik yang sudah dibuat, dapat diketahui demografi konsumen diketahui kota paling banyak yaitu **{most_common_state}**")

# Customer Demographic Order Status
st.subheader("Demografi Konsumen Berdasarkan Order Status")
common_status_ = order_status.value_counts().index[0]
st.markdown(f"Order Status Paling Banyak: **{common_status_}**")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=order_status.index,
            y=order_status.values,
            hue=order_status.index,
            order=order_status.index,
            palette=["#068DA9" if score == common_status else "#D3D3D3" for score in order_status.index]
            )

plt.title("Order Status", fontsize=15)
plt.xlabel("Status")
plt.ylabel("Jumlah")
plt.xticks(fontsize=12)
st.pyplot(fig)

with st.expander("Lihat Penjelasan"):
    st.write(f"Sesuai dengan grafik yang sudah dibuat, dapat diketahui order status paling banyak yaitu **{common_status_}**")

# Customer Demographic Geolocation
st.subheader("Demografi Konsumen")
map_plot.plot()

with st.expander("Lihat Penjelasan"):
    st.write(f"Sesuai dengan grafik yang sudah dibuat, dapat diketahui order status paling banyak yaitu **{common_status_}**")

# Footer
st.caption('Copyright (C) Picessa Kresna 2023')
