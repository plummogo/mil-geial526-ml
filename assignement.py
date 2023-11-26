import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Adatkészlet beolvasás
# Paramétere az elérési helye az adatkészletnek 
# Részletes leírás 'read_csv()' metódusról: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
data = pd.read_csv('dataset.csv')

# Adatkészlet tisztításának előkészítése
# 'price_card' és 'combo_type_id' oszlopok adatainak numerikus értékké alakítása
# errors='coerce' paraméter segítségével a nem numerikus adatok NaN értékre cseréli
# Részletes leírás 'to_numeric()' metódusról: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html
data['price_card'] = pd.to_numeric(data['price_card'], errors='coerce')
data['combo_type_id'] = pd.to_numeric(data['combo_type_id'], errors='coerce')

# Adatkészlet medián számítása
# 'price_card' és 'combo_type_id' oszlopok medián számítása
# Részletes leírás 'median()' metódusról: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.median.html
price_card_median = data['price_card'].median()
combo_type_id_median = data['combo_type_id'].median()

# Adatkészlet tisztítása
# 'price_card' és 'combo_type_id' adatainak feltöltése, ahol nem tudta az értékeket átalakítani számmá
# Részletes leírás 'fillna()' metódusról: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
data['price_card'].fillna(price_card_median, inplace=True)
data['combo_type_id'].fillna(combo_type_id_median, inplace=True)

# Adatkészlet leíró statisztika előkészítése
# 'price_card' és 'combo_type_id' adatainak 95%-os kvartilisének kiszámítása
# Paraméter a kvartilis értékének százalékos meghatározása 
# Részletes leírás 'quanitile()' metódusról: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html
price_card_quantile = data['price_card'].quantile(0.95)
all_time_score_quantile = data['all_time_score'].quantile(0.95)

# Adatkészlet értékeinek manipulációja kvartilis felsőérték alapján
# 'upper=quantile' paraméter segítségével minden értéket ami nagyobb a megadott paraméterként lecseréli arra az értékre
# Részletes leírás 'clip()' metódusról: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.clip.html
data['price_card'] = data['price_card'].clip(upper=price_card_quantile)
data['all_time_score'] = data['all_time_score'].clip(upper=all_time_score_quantile)

# Leíró statisztika konzolra való kiíratása
# Részletes leírás 'describe()' metódusról: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
print(data.describe())

# Kiugró értékek (Outliers) diagram vizualizációja
# Kiválasztottam az adatkészletből 4 numerikus értékeket tartalmazó oszlopokat
# Részletes leírás 'matplotlib' csomagról: https://matplotlib.org/stable/api/index.html 
columns = ['price_card', 'amount_card', 'all_time_score', 'ranking_player_tournament']
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=data[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# Korrelációs elemzés
# Adatkészletből kiszámítom a korrelációs mátrixot és hőtérképpel vizualizálom
# Adatkészletből kiválasztom azokat az oszlopokat amelyek numerikus értékeket tartalmaznak 'select_dtypes()' metódussal
# 'include=[np.number]' paraméterrel határozom meg hogy numerikus értékeket szűrjön
# Részletes leírás 'np.number' paraméterről: https://numpy.org/doc/stable/user/basics.types.html 
# Részletes leírás 'select_dtypes()' metódusról: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
# Részletes leírás 'corr()' metódusról: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
# 'annot=True' paraméter segítségével beállítom hogy a hőtérkép celláiban megjelenjenek értékek
# 'cmap=coolwarm' paraméter segítségével beállítom a hőtérkép színkódolását, hideg színeknél alacsonyabb értékek, melegebb színeknél magasabb értékek
# 'fmt=.2f' paraméter segítségével beállítom a hőtérkép értékeinek formátumát 2 tizedesjegyig 
# Részletes leírás 'heatmap()' metódusról: https://seaborn.pydata.org/generated/seaborn.heatmap.html
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()