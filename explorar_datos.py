import pandas as pd

df = pd.read_parquet("stuff_model_df.parquet")
diccionario = pd.read_csv("stuff_plus_data_dictionary.csv")

print("Dataset:")
print(df.shape)

print("\nDiccionario de variables:")
print(diccionario.head(20))

print("\nColumnas del diccionario:")
print(diccionario.columns.tolist())