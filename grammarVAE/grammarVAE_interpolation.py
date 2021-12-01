from rdkit.Chem import AllChem as Chem
import molecule_vae
import numpy as np

def getEquidistantPoints(p1, p2, parts):
    points = []
    for i in range(1, parts):
        points.append((p2-p1)/parts * i + p1)
    return np.array(points)

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
decoded_smiles = grammar_model.decode(getEquidistantPoints(z[0], z[1], 100))
failed_num = 0
for s in decoded_smiles:
    m = Chem.MolFromSmiles(s)
    if m is None:
        failed_num += 1
print(failed_num)
