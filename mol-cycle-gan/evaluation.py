from rdkit.Chem import AllChem as Chem
import numpy as np
import pandas as pd


def evaluate():
    dimention = 56
    interval_n = 2
    JTVAE = pd.read_csv('./data/input_data/rule_of_5/FDA_JTVAE.csv')
    grammer_enc = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20, 21,
                   22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39, 41,
                   42, 43, 44, 45, 46, 47, 48, 49]
    JTVAE = np.array(JTVAE)
    JTVAE_ = JTVAE[grammer_enc, 1:]
    JTVAE_ = JTVAE_[:10, :]
    smiles_encode = JTVAE[grammer_enc, 0]
    smiles_encode = smiles_encode[:10]
    n = JTVAE_.shape[0]
    print(n)
    print(smiles_encode)

    decoded_AB = np.array(pd.read_csv('./data/results/rule_of_5/smiles_list_n10_A_to_B.csv'))
    # decoded_BA = np.array(pd.read_csv('./data/results/rule_of_5/smiles_list_n10_B_to_A.csv'))
    # print(decoded_AB, decoded_BA)
    total_failed_number = 0
    total_not_identity = 0
    count = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            print("Processing {} pairs ...".format((i, j)))
            smiles = [smiles_encode[i], smiles_encode[j]]
            decoded_smiles_AB = decoded_AB[count*dimention*interval_n:(count+1)*dimention*interval_n]
            # decoded_smiles_BA = decoded_BA[count*dimention*interval_n:(count+1)*dimention*interval_n]
            # decoded_smiles = list(np.vstack((decoded_smiles_AB, decoded_smiles_BA))[:, 0])
            # print(decoded_smiles.shape)
            decoded_smiles = list(decoded_smiles_AB[:, 0])
            failed_num = 0
            success_cases = []
            for s in decoded_smiles:
                try:
                    m = Chem.MolFromSmiles(s)
                    if m is None:
                        failed_num += 1
                    else:
                        success_cases.append(m)
                except:
                    failed_num += 1
            print("Number of failed smiles:", failed_num, "/", len(decoded_smiles))
            print("Success rate:", float(len(decoded_smiles) - failed_num) / len(decoded_smiles))

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
            count += 1
    print("Total number of failed cases:", total_failed_number)
    print("Total number of not identity:", total_not_identity)
    print("Total number of decoded smiles:", decoded_AB.shape[0])

if __name__ == '__main__':
    evaluate()
