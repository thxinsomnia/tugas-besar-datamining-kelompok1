import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression


def plot_regresi_by_tempat(df, tempat_wisata, target='Wisnus'):
    data_tempat = df[df['Tempat Wisata'] == tempat_wisata]

    data_tahunan = data_tempat.groupby('Tahun')[target].sum().reset_index()
    X = data_tahunan['Tahun'].values.reshape(-1, 1)
    y = data_tahunan[target].values

    if len(X) < 2:
        print(f"Data untuk tempat wisata {tempat_wisata} kurang dari 2 tahun. Tidak dapat ditampilkan.")
        return

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.figure()
    plt.plot(data_tahunan['Tahun'], y, marker='o', label='Total Tahunan')
    plt.plot(data_tahunan['Tahun'], y_pred, 'r--', label='Regresi')
    plt.xlabel('Tahun')
    plt.ylabel(target)
    plt.title(f'Tren {target} untuk {tempat_wisata}')
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_regresi_by_wilayah(df, target='Wisnus'):
    wilayah_list = df['Wilayah'].unique()

    for wilayah in wilayah_list:
        data_wilayah = df[df['Wilayah'] == wilayah]
        data_tahunan = data_wilayah.groupby('Tahun')[target].sum().reset_index()

        X = data_tahunan['Tahun'].values.reshape(-1, 1)
        y = data_tahunan[target].values

        if len(X) < 2:
            continue  # skip wilayah with insufficient data

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        fig, ax = plt.subplots() # Ubah plt.figure() menjadi fig, ax = plt.subplots()
        line_data, = ax.plot(data_tahunan['Tahun'], y, marker='o', label='Total Tahunan') # Simpan objek Line2D
        line_regresi, = ax.plot(data_tahunan['Tahun'], y_pred, 'r--', label='Regresi') # Simpan objek Line2D

        ax.set_xlabel('Tahun')
        ax.set_ylabel(target)
        ax.set_title(f'Tren {target} Wilayah {wilayah}')
        ax.legend()
        plt.tight_layout()

        # --- Bagian Baru untuk Fungsionalitas Hover (Tooltip) ---

        # Buat objek annotation yang awalnya tidak terlihat
        # Pastikan annotation ini dibuat untuk setiap plot
        annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            # Posisi x dan y dari titik data yang di-hover
            x_data, y_data = line_data.get_data()
            x_val = x_data[ind["ind"][0]]
            y_val = y_data[ind["ind"][0]]

            annot.xy = (x_val, y_val) # Atur posisi annotation
            text = f"Tahun: {int(x_val)}\n{target}: {y_val:,.0f}" # Format teks tooltip
            annot.set_text(text) # Atur teks annotation

        def hover(event):
            # Periksa apakah event terjadi di atas sumbu (axis)
            if event.inaxes == ax:
                # Periksa apakah event terjadi di dekat titik data 'Total Tahunan'
                cont, ind = line_data.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle() # Gambar ulang kanvas
                else:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle() # Gambar ulang kanvas

        # Hubungkan event mouse motion dengan fungsi hover
        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show() # Tampilkan plot

def plot_tren_aggregat(df, target='Wisnus'):
    df_sum = df.groupby('Tahun')[target].sum().reset_index()

    X = df_sum['Tahun'].values.reshape(-1, 1)
    y = df_sum[target].values

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(8, 5))
    sns.lineplot(x='Tahun', y=target, data=df_sum, marker='o', label='Total Tahunan')
    plt.plot(df_sum['Tahun'], y_pred, 'r--', label='Regresi')
    plt.title(f"Tren Total {target} Seluruh Tempat Wisata")
    plt.xlabel("Tahun")
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        
def evaluate_regression_aggregate(df, target='Wisnus'):
    df_sum = df.groupby('Tahun')[target].sum().reset_index()

    X = df_sum['Tahun'].values.reshape(-1, 1)
    y = df_sum[target].values

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    print(f"Evaluasi Regresi untuk Total {target}")
    print(f"R² Score     : {r2:.2f}")
    print(f"MAE          : {mae:.2f}")
    print(f"MSE          : {mse:.2f}")
    print(f"RMSE         : {rmse:.2f}")
