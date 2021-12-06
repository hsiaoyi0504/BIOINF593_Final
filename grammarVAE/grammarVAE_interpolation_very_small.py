from rdkit.Chem import AllChem as Chem
import molecule_vae
import numpy as np
import pandas as pd
import pickle

def getEquidistantPoints(p1, p2, parts):
    points = []
    for i in range(1, parts):
        points.append((p2-p1)/parts * i + p1)
    points = np.array(points)
    # print(points.shape)
    return points

def getByDimensionalInterpolationPoints(p1, p2, parts=2):
    points = []
    num_dims = p1.shape[0]
    for i in range(num_dims):
        for j in range(1, parts):
            point = np.copy(p1)
            point[i] = ((p2-p1)/parts * j)[i] + p1[i]
            points.append(point)
            point = np.copy(p2)
            point[i] = ((p1-p2)/parts * j)[i] + p2[i]
            points.append(point)
    points = np.array(points)
    return points

# 1. load grammar VAE
grammar_weights = "./pretrained/zinc_vae_grammar_L56_E100_val.hdf5"
grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)

# 2. let's encode and decode some example SMILES strings
smiles = [
        "O=C([O-])[C@@H]1CC2=c3ccccc3=[NH+][C@H]2[C@H](C(=O)[O-])[NH2+]1", # smallest logP
        "CC[N+]1(CC)c2cc(/N=C/c3ccccc3)c(/C=C/c3ccccc3)cc2C[C@H]1c1ccccc1" # largest logP
        ]

# z: encoded latent points
# NOTE: this operation returns the mean of the encoding distribution
# if you would like it to sample from that distribution instead
# replace line 83 in molecule_vae.py with: return self.vae.encoder.predict(one_hot)
z = grammar_model.encode(smiles)
# print(getEquidistantPoints(z[0], z[1], 100))

# mol: decoded SMILES string
# NOTE: decoding is stochastic so calling this function many
# times for the same latent point will return different answers
# for mol,real in zip(grammar_model.decode(z1),smiles):
#    print mol + '  ' + real
# print(grammar_model.decode(getEquidistantPoints(z[0], z[1], 100)))
decoded_smiles = grammar_model.decode(getEquidistantPoints(z[0], z[1], 101))
failed_num = 0
for s in decoded_smiles:
    m = Chem.MolFromSmiles(s)
    if m is None:
        failed_num += 1
print(failed_num)

decoded_smiles = grammar_model.decode(getByDimensionalInterpolationPoints(z[0], z[1]))
failed_num = 0
success_cases = []
for s in decoded_smiles:
    m = Chem.MolFromSmiles(s)
    if m is None:
        failed_num += 1
    else:
        success_cases.append(m)
print(failed_num)
print(len(decoded_smiles))
print("Success rate:", float(len(decoded_smiles)-failed_num)/len(decoded_smiles))
rdkit_smiles = []
for s in smiles:
    rdkit_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
not_identity = 0
for m in success_cases:
    if Chem.MolToSmiles(m) not in rdkit_smiles:
        not_identity += 1
print("Not identity:", not_identity)

smiles = [
        "O=C([O-])[C@@H]1CC2=c3ccccc3=[NH+][C@H]2[C@H](C(=O)[O-])[NH2+]1", # smallest logP
        "CC[N+]1(CC)c2cc(/N=C/c3ccccc3)c(/C=C/c3ccccc3)cc2C[C@H]1c1ccccc1" # largest logP
        ]

df = pd.read_csv("../FDA_JTVAE.csv")
selected_fda_smiles = df["smiles"].tolist()[:50]
# smiles_pairs = []
# for i in range(len(selected_fda_smiles)):
#     for j in range(i+1,len(selected_fda_smiles)):
#        smiles_pairs.append([selected_fda_smiles[i], selected_fda_smiles[j]])

success_encode = []
success_index = []
for i, s in enumerate(selected_fda_smiles):
    try:
        z = grammar_model.encode([s])
        success_encode.append(s)
        success_index.append(i)
    except:
        continue
    # except IndexError:
    #    continue
    #except StopIteration:
    #    continue
print("Success index (", len(success_index) , "):", success_index)

success_encode = success_encode[:10]

z = grammar_model.encode(success_encode)

input_smiles_pairs = []
result = []
total_failed_number = 0
total_not_identity = 0
# for num, pair in enumerate(smiles_pairs):
for i in range(len(success_encode)):
    for j in range(i+1,len(success_encode)):
        print("Processing {} pairs ...".format((i,j)))
        smiles = [success_encode[i], success_encode[j]] 
        decoded_smiles = grammar_model.decode(getByDimensionalInterpolationPoints(z[i], z[j]))
        input_smiles_pairs.append(smiles)
        result.append(decoded_smiles)
        failed_num = 0
        success_cases = []
        for s in decoded_smiles:
            m = Chem.MolFromSmiles(s)
            if m is None:
                failed_num += 1
            else:
                success_cases.append(m)
        print("Number of failed smiles:", failed_num, "/", len(decoded_smiles))
        print("Success rate:", float(len(decoded_smiles)-failed_num)/len(decoded_smiles))
        rdkit_smiles = []
        for s in smiles:
            rdkit_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
        not_identity = 0
        for m in success_cases:
            if Chem.MolToSmiles(m) not in rdkit_smiles:
                not_identity += 1
        print("Number of not identity:", not_identity)
        total_failed_number += failed_num
        total_not_identity += not_identity

print("Total number of failed cases:", total_failed_number)
print("Total number of not identity:", total_not_identity)
print("Total number of decoded smiles:", len(success_encode) * (len(success_encode) - 1 ) * 2 * z[0].shape[0])
with open("../grammarVAE_result_very_small.pkl", "wb") as f:
    pickle.dump([success_encode, input_smiles_pairs, result], f)

