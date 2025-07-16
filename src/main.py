from data_loader import load_data
from model import analisis_tren_regresi

if __name__ == "__main__":
    df = load_data("data/processed/data_wisata_cleaned_filtered.csv")

    tren_wisnus = analisis_tren_regresi(df, target='Wisnus')
    tren_wisman = analisis_tren_regresi(df, target='Wisman')

    tren_wisnus.to_csv('hasil_tren_wisnus.csv', index=False)
    tren_wisman.to_csv('hasil_tren_wisman.csv', index=False)

    print("Analisis selesai. Hasil disimpan.")
