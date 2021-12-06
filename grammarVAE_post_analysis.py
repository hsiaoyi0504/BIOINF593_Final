import pickle

with open("grammarVAE_result_very_small.pkl", "rb") as f:
    success_encode, input_smiles_pairs, result = pickle.load(f)
print(len(success_encode))
print(len(input_smiles_pairs))
print(len(input_smiles_pairs[0]))
print(len(result))
print(len(result[0]))
