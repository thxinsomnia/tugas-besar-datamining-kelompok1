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

        plt.figure()
        plt.plot(data_tahunan['Tahun'], y, marker='o', label='Total Tahunan')
        plt.plot(data_tahunan['Tahun'], y_pred, 'r--', label='Regresi')
        plt.xlabel('Tahun')
        plt.ylabel(target)
        plt.title(f'Tren {target} Wilayah {wilayah}')
        plt.legend()
        plt.tight_layout()
        plt.show()


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
