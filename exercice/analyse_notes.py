import pandas as pd

df = pd.read_csv("data/notes_eleves.csv")

print("\naperçu des premières lignes:")
print(df.head())

print("informations en general sur le dataframe notes-eleves:")
df.info()

print("\ndescription du dataframe notes-eleves:")
print(df.describe())

print("\nvaleurs manquantes:")
print(df.isnull().sum())
