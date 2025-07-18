import pandas as pd

def load_clean_data():
    df = pd.read_csv('../data/processed/data_wisata_marge.csv')

    # Filter tahun 2020–2024
    df = df[df['Tahun'].between(2020, 2024)]

    # Bersihkan data NaN jika ada
    df = df.dropna(subset=['Wilayah', 'Tempat Wisata', 'Tahun', 'Wisnus', 'Wisman'])

    # Tambahkan kolom jumlah total wisatawan
    df['Jumlah'] = df['Wisnus'] + df['Wisman']

    return df
