# S1-partie-1
import numpy as np
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


#############################################
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


#############################################
# S1-partie-4
print("moyenne generale par etudiant")


def moyenne_ponderee(group):
    return np.dot(group["note"], group["coefficient"]) / np.sum(group["coefficient"])


moy_etudiant = df.groupby("eleve_id").apply(moyenne_ponderee)
print(moy_etudiant)

##############################################
# S1-partie-5

moy_etudiant = moy_etudiant.reset_index()
moy_etudiant.columns = ["eleve_id", "moyenne"]
df_moyenne = pd.merge(df, moy_etudiant, on="eleve_id")

# ajout de la colonne admis avec np.where
df_moyenne["admis"] = np.where(df_moyenne["moyenne"] >= 10, "Oui", "Non")


# ajout de la colonne mention avec np.select
conditions = [
    df_moyenne["moyenne"] >= 16,
    df_moyenne["moyenne"] >= 14,
    df_moyenne["moyenne"] >= 12,
    df_moyenne["moyenne"] >= 10,
    df_moyenne["moyenne"] < 10,
]
choix = ["Très Bien", "Bien", "Assez Bien", "passable", "non attribué"]
df_moyenne["mention"] = np.select(conditions, choix, default="non attribué")


######################################
# S1-partie-6
# classement_finale.csv
df_moyenne[
    ["eleve_id", "nom", "prenom", "classe", "moyenne", "admis", "mention"]
].drop_duplicates().sort_values("moyenne", ascending=False).to_csv(
    "output/classement_final.csv", index=False
)

# eleves_en_difficulte.csv
df_moyenne[df_moyenne["admis"] == "Non"][
    ["eleve_id", "nom", "prenom", "classe", "moyenne"]
].drop_duplicates().to_csv("output/eleves_en_difficulte.csv", index=False)


# moyennes_par_matiere.json
df.groupby("matiere")["note"].mean().to_json(
    "output/moyennes_par_matiere.json", force_ascii=False
)
