import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def plot_regresi_tempat(df, nama_tempat, target='Wisnus'):
    data = df[df['Tempat Wisata'].str.lower() == nama_tempat.lower()].sort_values('Tahun')
    if data.empty:
        print(f"Tidak ada data untuk: {nama_tempat}")
        return

    X = data['Tahun'].values.reshape(-1, 1)
    y = data[target].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data['Tahun'], y=y, color='blue', s=100, label='Data Aktual')
    sns.lineplot(x=data['Tahun'], y=y_pred, color='red', label='Garis Tren')
    plt.title(f"{target} - Regresi Linear: {nama_tempat}")
    plt.xlabel("Tahun")
    plt.ylabel("Jumlah Pengunjung")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
