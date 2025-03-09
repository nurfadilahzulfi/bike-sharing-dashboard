import streamlit as st
import numpy as np
import pandas as pd
# menambahkan konfigurasi matplotlib dengan backend Agg sebelum impor plt
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-interactive Agg
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide"
)

# Fungsi untuk load dan persiapkan data
@st.cache_data
def load_data():
    # Memuat tabel day
    day_df = pd.read_csv("dashboard/day_clean.csv")
    
    # Memuat tabel hour
    hour_df = pd.read_csv("dashboard/hour_clean.csv")
    
    # Menghapus kolom workingday
    hour_df.drop(['workingday'], axis=1, inplace=True)
    day_df.drop(['workingday'], axis=1, inplace=True)
    
    # Mengubah tipe data menjadi category
    columns = ['season', 'mnth', 'holiday', 'weekday', 'weathersit']
    for column in columns:
        day_df[column] = day_df[column].astype("category")
        hour_df[column] = hour_df[column].astype("category")
    
    # Menghandling tipe data dteday menjadi datetime
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    
    # Mengganti nama kolom agar lebih mudah dibaca
    day_df.rename(columns={
        'yr': 'year',
        'mnth': 'month',
        'weekday': 'one_of_week',
        'weathersit': 'weather_situation',
        'windspeed': 'wind_speed',
        'cnt': 'count_cr',
        'hum': 'humidity'
    }, inplace=True)
    
    hour_df.rename(columns={
        'yr': 'year',
        'hr': 'hours',
        'mnth': 'month',
        'weekday': 'one_of_week',
        'weathersit': 'weather_situation',
        'windspeed': 'wind_speed',
        'cnt': 'count_cr',
        'hum': 'humidity'
    }, inplace=True)
    
    # For season replacement
    day_df['season'] = day_df['season'].replace((1,2,3,4), ('Spring','Summer','Fall','Winter'))
    hour_df['season'] = hour_df['season'].replace((1,2,3,4), ('Spring','Summer','Fall','Winter'))
    
    # For month replacement
    month_mapping = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    day_df['month'] = day_df['month'].replace(month_mapping)
    hour_df['month'] = hour_df['month'].replace(month_mapping)
    
    # For weather_situation replacement
    weather_mapping = {1: 'Clear', 2: 'Misty', 3: 'Light_rainsnow', 4: 'Heavy_rainsnow'}
    day_df['weather_situation'] = day_df['weather_situation'].replace(weather_mapping)
    hour_df['weather_situation'] = hour_df['weather_situation'].replace(weather_mapping)
    
    # For one_of_week replacement
    weekday_mapping = {
        0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
        4: 'Thursday', 5: 'Friday', 6: 'Saturday'
    }
    day_df['one_of_week'] = day_df['one_of_week'].replace(weekday_mapping)
    hour_df['one_of_week'] = hour_df['one_of_week'].replace(weekday_mapping)
    
    # For year replacement
    day_df['year'] = day_df['year'].replace({0: '2011', 1: '2012'})
    hour_df['year'] = hour_df['year'].replace({0: '2011', 1: '2012'})
    
    # Membuat kolom category_days
    def get_category_days(one_of_week):
        if one_of_week in ["Saturday", "Sunday"]:
            return "weekend"
        else:
            return "weekdays"
    
    hour_df["category_days"] = hour_df["one_of_week"].apply(get_category_days)
    day_df["category_days"] = day_df["one_of_week"].apply(get_category_days)
    
    # Membuat kategori kelembaban
    def classify_humidity(humidity):
        if humidity < 45:
            return "Terlalu kering"
        elif humidity >= 45 and humidity < 65:
            return "Ideal"
        else:
            return "Terlalu Lembab"
    
    hour_df["humidity_category"] = hour_df["humidity"].apply(classify_humidity)
    day_df["humidity_category"] = day_df["humidity"].apply(classify_humidity)
    
    return day_df, hour_df

# Fungsi untuk membuat visualisasi
def get_total_count_by_hour_df(hour_df):
    hour_count_df = hour_df.groupby(by="hours").agg({
        "count_cr": ["sum"]
    })
    hour_count_df.columns = ["total_count"]
    return hour_count_df.reset_index()

def get_total_count_by_season(day_df):
    return day_df.groupby(by="season").count_cr.sum().sort_values(ascending=False).reset_index()

def get_total_count_by_weather(hour_df):
    return hour_df.groupby(by="weather_situation").count_cr.sum().sort_values(ascending=False).reset_index()

def get_total_by_category_days(day_df):
    return day_df.groupby(by="category_days").count_cr.sum().reset_index()

def get_total_by_humidity(hour_df):
    return hour_df.groupby(by="humidity_category").agg({
        "count_cr": ["sum"]
    }).reset_index()

def get_monthly_trend(day_df):
    monthly_counts = day_df.groupby(pd.Grouper(key='dteday', freq='M')).agg({
        'count_cr': 'sum'
    }).reset_index()
    return monthly_counts

def get_customer_segmentation(day_df):
    total_casual = day_df['casual'].sum()
    total_registered = day_df['registered'].sum()
    return pd.DataFrame({
        'tipe_pelanggan': ['Casual', 'Registered'],
        'jumlah': [total_casual, total_registered]
    })

# Load data
day_df, hour_df = load_data()

# Header dashboard
st.title("🚲 Dashboard Bike Sharing")
st.markdown("Dashboard untuk menganalisa data penyewaan sepeda")

# Membuat sidebar untuk filter
st.sidebar.header("Filter Data")

# Filter berdasarkan tahun
selected_year = st.sidebar.selectbox("Pilih Tahun", ['Semua', '2011', '2012'])
if selected_year != 'Semua':
    day_filtered = day_df[day_df['year'] == selected_year]
    hour_filtered = hour_df[hour_df['year'] == selected_year]
else:
    day_filtered = day_df
    hour_filtered = hour_df

# Filter berdasarkan musim
selected_season = st.sidebar.selectbox("Pilih Musim", ['Semua', 'Spring', 'Summer', 'Fall', 'Winter'])
if selected_season != 'Semua':
    day_filtered = day_filtered[day_filtered['season'] == selected_season]
    hour_filtered = hour_filtered[hour_filtered['season'] == selected_season]

# Filter berdasarkan hari
selected_day_type = st.sidebar.selectbox("Tipe Hari", ['Semua', 'weekdays', 'weekend'])
if selected_day_type != 'Semua':
    day_filtered = day_filtered[day_filtered['category_days'] == selected_day_type]
    hour_filtered = hour_filtered[hour_filtered['category_days'] == selected_day_type]

# Menampilkan metrics di bagian atas
col1, col2, col3, col4 = st.columns(4)

# Menampilkan total penyewaan
with col1:
    total_rentals = int(day_filtered['count_cr'].sum())
    st.metric("Total Penyewaan", f"{total_rentals:,}")

# Menampilkan rata-rata penyewaan per hari
with col2:
    avg_rentals_per_day = int(day_filtered['count_cr'].mean())
    st.metric("Rata-rata Penyewaan per Hari", f"{avg_rentals_per_day:,}")

# Menampilkan hari dengan penyewaan tertinggi
with col3:
    max_day = day_filtered.loc[day_filtered['count_cr'].idxmax()]
    st.metric("Penyewaan Tertinggi", f"{int(max_day['count_cr']):,}", 
              f"Tanggal {max_day['dteday'].strftime('%d %b %Y')}")

# Menampilkan jumlah hari dalam dataset
with col4:
    total_days = day_filtered.shape[0]
    st.metric("Jumlah Hari", f"{total_days:,}")

# Membuat visualisasi dengan tabs
tab1, tab2, tab3, tab4 = st.tabs(["Penyewaan per Jam", "Penyewaan per Musim", 
                                  "Trend Bulanan", "Segmentasi Pelanggan"])

# Tab 1: Penyewaan per Jam
with tab1:
    st.header("Penyewaan Berdasarkan Jam")
    hour_count_df = get_total_count_by_hour_df(hour_filtered)
    
    # Membuat visualisasi penyewaan per jam
    # Perbaikan 2: Memastikan figure disimpan dengan benar
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="hours", y="total_count", data=hour_count_df, ax=ax)
    ax.set_title("Jumlah Penyewaan Berdasarkan Jam")
    ax.set_xlabel("Jam")
    ax.set_ylabel("Jumlah Penyewaan")
    st.pyplot(fig)  # Pastikan objek fig diteruskan ke st.pyplot()
    
    # Menampilkan jam tersibuk dan tersepi
    col1, col2 = st.columns(2)
    with col1:
        busy_hour = hour_count_df.loc[hour_count_df['total_count'].idxmax()]
        st.metric("Jam Tersibuk", f"{int(busy_hour['hours'])}:00", f"{int(busy_hour['total_count']):,} penyewaan")
    
    with col2:
        quiet_hour = hour_count_df.loc[hour_count_df['total_count'].idxmin()]
        st.metric("Jam Tersepi", f"{int(quiet_hour['hours'])}:00", f"{int(quiet_hour['total_count']):,} penyewaan")

# Tab 2: Penyewaan per Musim
with tab2:
    st.header("Penyewaan Berdasarkan Musim")
    season_df = get_total_count_by_season(day_filtered)
    
    # Membuat visualisasi penyewaan per musim
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x="season", y="count_cr", data=season_df, ax=ax)
    ax.set_title("Jumlah Penyewaan Berdasarkan Musim")
    ax.set_xlabel("Musim")
    ax.set_ylabel("Jumlah Penyewaan")
    
    # Menambahkan label pada bar
    for p in bars.patches:
        bars.annotate(format(int(p.get_height()), ','), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    
    st.pyplot(fig)  # Pastikan objek fig diteruskan ke st.pyplot()
    
    # Menampilkan perbandingan musim
    col1, col2 = st.columns(2)
    
    # Visualisasi berdasarkan cuaca
    with col1:
        st.subheader("Berdasarkan Cuaca")
        weather_df = get_total_count_by_weather(hour_filtered)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="weather_situation", y="count_cr", data=weather_df, ax=ax)
        ax.set_title("Jumlah Penyewaan Berdasarkan Cuaca")
        ax.set_xlabel("Cuaca")
        ax.set_ylabel("Jumlah Penyewaan")
        plt.xticks(rotation=45)
        st.pyplot(fig)  # Pastikan objek fig diteruskan ke st.pyplot()
    
    # Visualisasi berdasarkan tipe hari
    with col2:
        st.subheader("Berdasarkan Tipe Hari")
        day_type_df = get_total_by_category_days(day_filtered)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="category_days", y="count_cr", data=day_type_df, ax=ax)
        ax.set_title("Jumlah Penyewaan Berdasarkan Tipe Hari")
        ax.set_xlabel("Tipe Hari")
        ax.set_ylabel("Jumlah Penyewaan")
        st.pyplot(fig)  # Pastikan objek fig diteruskan ke st.pyplot()

# Tab 3: Trend Bulanan
with tab3:
    st.header("Trend Penyewaan Bulanan")
    monthly_data = get_monthly_trend(day_filtered)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly_data['dteday'], monthly_data['count_cr'], marker='o', linestyle='-')
    ax.set_title("Trend Penyewaan Sepeda Bulanan")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah Penyewaan")
    plt.xticks(rotation=45)
    st.pyplot(fig)  # Pastikan objek fig diteruskan ke st.pyplot()
    
    # Menampilkan data tren dalam bentuk tabel
    st.subheader("Data Bulanan")
    monthly_data['bulan'] = monthly_data['dteday'].dt.strftime('%B %Y')
    monthly_display = monthly_data[['bulan', 'count_cr']].rename(columns={
        'bulan': 'Bulan', 
        'count_cr': 'Jumlah Penyewaan'
    })
    st.dataframe(monthly_display)

# Tab 4: Segmentasi Pelanggan
with tab4:
    st.header("Segmentasi Pelanggan")
    customer_seg = get_customer_segmentation(day_filtered)
    
    col1, col2 = st.columns(2)
    
    # Pie chart untuk tipe pelanggan
    with col1:
        st.subheader("Proporsi Tipe Pelanggan")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(customer_seg['jumlah'], labels=customer_seg['tipe_pelanggan'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        st.pyplot(fig)  # Pastikan objek fig diteruskan ke st.pyplot()
    
    # Barplot untuk tipe pelanggan
    with col2:
        st.subheader("Jumlah Pelanggan per Tipe")
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.barplot(x='tipe_pelanggan', y='jumlah', data=customer_seg, ax=ax)
        ax.set_title("Jumlah Pelanggan Berdasarkan Tipe")
        ax.set_xlabel("Tipe Pelanggan")
        ax.set_ylabel("Jumlah")
        
        # Menambahkan label pada bar
        for p in ax.patches:
            ax.annotate(format(int(p.get_height()), ','), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')
        
        st.pyplot(fig)  # Pastikan objek fig diteruskan ke st.pyplot()

# Tampilkan dataset asli di bagian bawah
st.header("Dataset Asli")
show_data = st.checkbox("Tampilkan Dataset")

if show_data:
    tab_day, tab_hour = st.tabs(["Data Harian", "Data Per Jam"])
    
    with tab_day:
        st.dataframe(day_df)
    
    with tab_hour:
        st.dataframe(hour_df)