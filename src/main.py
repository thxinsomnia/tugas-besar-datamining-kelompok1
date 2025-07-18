from data_loader import load_clean_data
from model import analisis_tren_regresi
from utils import plot_regresi_by_tempat, evaluate_regression_aggregate, plot_tren_aggregat
from utils import plot_regresi_by_wilayah


# Load data
df = load_clean_data()

# Analisis tren masing-masing tempat wisata
tren_wisnus = analisis_tren_regresi(df, target='Wisnus')
tren_wisman = analisis_tren_regresi(df, target='Wisman')
plot_regresi_by_tempat(df, tempat_wisata='Museum Bali', target='Wisnus')
plot_regresi_by_tempat(df, tempat_wisata='Museum Bali', target='Wisman')

print("Tren Wisnus:")
print(tren_wisnus.head())
print("\nTren Wisman:")
print(tren_wisman.head())

# Visualisasi berdasarkan wilayah (fix: ini sebelumnya belum dipanggil)
plot_regresi_by_wilayah(df, target='Wisnus')
plot_regresi_by_wilayah(df, target='Wisman')

# Visualisasi agregat + evaluasi seluruh data
plot_tren_aggregat(df, target='Wisnus')
evaluate_regression_aggregate(df, target='Wisnus')

plot_tren_aggregat(df, target='Wisman')
evaluate_regression_aggregate(df, target='Wisman')
