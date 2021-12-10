import csv
import pandas as pd


def separate_250k():
    smiles = pd.read_csv('./X_JTVAE_250k_rndm_zinc.csv')
    train_n = 100000
    test_n = 20000
    train = smiles.loc[:train_n-1]
    test = smiles.loc[train_n:train_n+test_n-1]
    train.to_csv('./input_rof/rof_JTVAE_train_B.csv', index=False)
    test.to_csv('./input_rof/rof_JTVAE_test_B.csv', index=False)


def organize_rof():
    with open('./rule_of_5_violate_smiles.txt') as f:
        smiles_all = f.readlines()
        print(len(smiles_all))

    with open('./rof_encoded/encodable_final.txt') as f:
        encodable = f.readlines()
        encodable = [int(float(x.strip())) for x in encodable]
        print(len(encodable))
        encodable_smiles = []
        for j in range(len(encodable)):
            if encodable[j] == 1:
                encodable_smiles.append(smiles_all[j].strip())
        print(len(encodable_smiles))

    with open('./rof_encoded/latent_features_final.txt') as f:
        JTVAE = f.readlines()
        print(len(JTVAE))
        rof_JTVAE = []
        for i in range(len(JTVAE)):
            z = JTVAE[i].strip()
            ls = z.split()
            ls = [float(x) for x in ls]
            ls = [encodable_smiles[i]] + ls
            rof_JTVAE.append(ls)
        print(rof_JTVAE[0])
    indexes = ['smiles']
    for d in range(56):
        indexes.append('jtvae_Id_'+str(d))
    print(indexes)

    # all: 179107
    # 80%: 143285 - 100000
    # 20%: 35822 - 20000
    train_n = 100000
    test_n = 20000

    with open('./rof_encoded/rof_JTVAE_A.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(indexes)
        writer.writerows(rof_JTVAE)

    with open('./rof_encoded/rof_JTVAE_train_A.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(indexes)
        writer.writerows(rof_JTVAE[:train_n])
    
    with open('./rof_encoded/rof_JTVAE_test_A.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(indexes)
        writer.writerows(rof_JTVAE[train_n:train_n+test_n])


def count_violation():
    count_2 = 0
    count_3 = 0
    count_4 = 0
    with open('./rule_of_5_violate_count.txt') as f:
        total = f.readlines()
        print(len(total))

    with open('./rof_encoded/encodable_final.txt') as f:
        encodable = f.readlines()
        encodable = [int(float(x.strip())) for x in encodable]
        print(len(encodable))
        encodable_violate = []
        count = 0
        for j in range(len(encodable)):
            if encodable[j] == 1 and count < 100000:
                count += 1
                # encodable_violate.append(total[j].strip())
                vio = total[j].strip()
                if vio == '2':
                    count_2 += 1
                if vio == '3':
                    count_3 += 1
                if vio == '4':
                    count_4 += 1
    print(count_2, count_3, count_4)


if __name__ == '__main__':
    count_violation()
    # separate_250k()
    # organize_rof()