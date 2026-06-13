# S1-partie-1

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


# S1-partie-2
# Filtre 1 : élèves avec note >= 15 en classe 3ème A
print("élèves avec note >= 15 en classe 3ème A :")
print(df[(df["note"] >= 15) & (df["classe"] == "3ème A")])

# Filtre 2 : lignes où nom = Sassi, colonnes spécifiques avec loc
print("\nélèves Sassi — colonnes sélectionnées :")
print(df.loc[df["nom"] == "Sassi", ["nom", "prenom", "matiere", "note"]])

# Filtre 3 : matières avec isin
print("\nfiltrage par matières :")
print(df[df["matiere"].isin(["Anglais", "Français", "Informatique"])])


# S1-partie-3

# Moyenne par matière triée
print("moyenne par matière (ordre décroissant):")
print(df.groupby("matiere")["note"].mean().sort_values(ascending=False))

# Moyenne par classe
print("\nmoyenne par classe:")
print(df.groupby("classe")["note"].mean())

# Note max et min par élève
print("\nnote max et min par élève:")
print(df.groupby("eleve_id").agg(note_max=("note", "max"), note_min=("note", "min")))