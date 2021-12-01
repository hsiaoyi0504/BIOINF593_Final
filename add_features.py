import pandas as pd
from rdkit.Chem import Descriptors, PandasTools

df = pd.read_csv("./mol-cycle-gan/data/input_data/250k_rndm_zinc_drugs_clean_3_canonized.csv")

PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles', molCol='Molecule')
df["molecular_weight"] = df['Molecule'].map(lambda x: Descriptors.ExactMolWt(x))
df["h_bond_donor"] = df['Molecule'].map(lambda x: Descriptors.NumHDonors(x))
df["h_bond_acceptors"] = df['Molecule'].map(lambda x: Descriptors.NumHAcceptors(x))

def rule_of_five(row):
    num = 0
    # print(row)
    if row["molecular_weight"] >= 500:
        num += 1
    if row["h_bond_donor"] > 5:
        num += 1
    if row["h_bond_acceptors"] > 10:
        num += 1
    if row["logP"] >= 5:
        num += 1
    return num

df['rule_of_5'] = df.apply(lambda row: rule_of_five(row), axis=1)
df.drop(columns=['Molecule']).to_csv("./mol-cycle-gan/data/input_data/250k_zinc_with_features.csv", index=False)

