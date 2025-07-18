import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def analisis_tren_regresi(df, target='Wisnus', threshold=100):
    hasil = []

    for tempat in df['Tempat Wisata'].unique():
        data = df[df['Tempat Wisata'] == tempat].sort_values('Tahun')
        X = data['Tahun'].values.reshape(-1, 1)
        y = data[target].values

        if np.all(y == 0):
            continue

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]

        if slope > threshold:
            tren = 'Naik'
        elif slope < -threshold:
            tren = 'Turun'
        else:
            tren = 'Stabil'

        hasil.append({
            'Tempat Wisata': tempat,
            'Tren': tren,
            'Slope': round(slope, 2),
            'Target': target
        })

    return pd.DataFrame(hasil)
