import pickle
import pandas as pd
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors

df_250k = pd.read_csv("./mol-cycle-gan/data/input_data/250k_rndm_zinc_drugs_clean_3_canonized.csv")
smiles_250k = df_250k["smiles"]
smiles_250k = [ Chem.MolToSmiles(Chem.MolFromSmiles(s), 1) for s in smiles_250k ]
molcycle_train = pd.read_csv("rof_JTVAE_train_A.csv")
molcycle_train_smiles = molcycle_train["smiles"].tolist()
molcycle_train_smiles = [ Chem.MolToSmiles(Chem.MolFromSmiles(s), 1) for s in molcycle_train_smiles]



# with open("grammarVAE_result_very_small.pkl", "rb") as f: # 25
with open("grammarVAE_result_small.pkl", "rb") as f:  # 10
    success_encode, input_smiles_pairs, result_grammar = pickle.load(f)

df_mol_cycle_1 = pd.read_csv("smiles_list_n25_16800_A_to_B.csv")
df_mol_cycle_2 = pd.read_csv("smiles_list_n25_33600_A_to_B.csv")
result_mol_cycle = df_mol_cycle_1['SMILES'].tolist() + df_mol_cycle_2['SMILES'].tolist()
# print(len(set(result_mol_cycle)))

# print(result_mol_cycle[:2])
# print(success_encode)
# print(len(success_encode))
# print(len(input_smiles_pairs))
# print(len(input_smiles_pairs[0]))
# print(len(result_grammar))
# print(len(result_grammar[0]))
# print(len(result_mol_cycle))

def get_success_cases(decoded_smiles):
    success_cases = []
    failed_num = 0
    for s in decoded_smiles:
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                failed_num += 1
            else:
                # success_cases.append(m)
                success_cases.append(s)
        except:
            failed_num += 1
    return (success_cases, failed_num)

def get_not_identity_num(smiles, check_list):
# def get_not_identity_num(mols, check_list):
    check_list_smiles = []
    for s in check_list:
        check_list_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(s),1))
    # check_list_smiles = check_list
    not_identity_smiles = []
    not_identity_num = 0
    # for m in mols:
    #    smiles = Chem.MolToSmiles(m, 1)
    #    if smiles not in check_list_smiles:
    for s in smiles:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(s), 1)
        if smi not in check_list_smiles:
            not_identity_num += 1
            not_identity_smiles.append(s)
    return (not_identity_num, not_identity_smiles)

def get_pass_rule_of_five_num(smiles):
# def get_pass_rule_of_five_num(mols):
    pass_num = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
    # for m in mols:
        mw = Descriptors.ExactMolWt(m)
        if mw >= 500:
            continue
        h_bond_donor = Descriptors.NumHDonors(m)
        if h_bond_donor > 5:
            continue
        h_bond_acceptors = Descriptors.NumHAcceptors(m)
        if h_bond_acceptors > 10:
            continue
        logP = Descriptors.MolLogP(m)
        if logP >= 5:
            continue
        pass_num += 1
    return pass_num


def check_novelity(smiles_list, check_list):
    novelty_num = 0
    for s in smiles_list:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(s), 1)
        if smi not in check_list:
        # if s not in check_list:
            novelty_num += 1
    return novelty_num


grammar_total_failed_number = 0
grammar_total_not_identity = 0
grammar_total_rule_of_5 = 0
grammar_total_unique = 0
grammar_total_novelty_num = 0
max_grammar_unique = 0
molcycle_total_failed_number = 0
molcycle_total_not_identity = 0
molcycle_total_rule_of_5 = 0
molcycle_total_unique = 0
molcycle_total_novelty_num = 0
max_molcycle_unique = 0
for i, s in enumerate(input_smiles_pairs):
    # print(input_smiles_pairs[i])
    # Check sucess cases
    grammar_success_cases, grammar_failed_num = get_success_cases(result_grammar[i])
    molcycle_success_cases, molcycle_failed_num = get_success_cases(result_mol_cycle[i*112:(i+1)*112])
    grammar_total_failed_number += grammar_failed_num
    molcycle_total_failed_number += molcycle_failed_num
    # Not identity
    grammar_not_identity_num, grammar_not_identity_smiles = get_not_identity_num(grammar_success_cases, input_smiles_pairs[i])
    molcycle_not_identity_num, molcycle_not_identity_smiles = get_not_identity_num(molcycle_success_cases, input_smiles_pairs[i])
    grammar_total_not_identity += grammar_not_identity_num
    molcycle_total_not_identity += molcycle_not_identity_num
    grammar_not_identity_smiles = [str(smi) for smi in grammar_not_identity_smiles]
    molcycle_not_identity_smiles = [str(smi) for smi in molcycle_not_identity_smiles]
    grammar_unique_smiles = list(set(grammar_not_identity_smiles))
    molcycle_unique_smiles = list(set(molcycle_not_identity_smiles))
    num_grammar_unique = len(grammar_unique_smiles)
    num_molcycle_unique = len(molcycle_unique_smiles)
    grammar_total_unique += num_grammar_unique
    molcycle_total_unique += num_molcycle_unique
    if num_grammar_unique > max_grammar_unique and num_molcycle_unique > max_molcycle_unique:
        selected_pair = s
        max_grammar_unique = num_grammar_unique
        max_molcycle_unique = num_molcycle_unique
        selected_grammar_generation = grammar_unique_smiles
        selected_molcycle_generation = molcycle_unique_smiles
    # Rule of 5
    grammar_rule_of_5 = get_pass_rule_of_five_num(grammar_success_cases)
    molcycle_rule_of_5 = get_pass_rule_of_five_num(molcycle_success_cases)
    grammar_total_rule_of_5 += grammar_rule_of_5
    molcycle_total_rule_of_5 += molcycle_rule_of_5
    grammar_novelty_num = check_novelity(grammar_unique_smiles, smiles_250k)
    molcycle_novelty_num = check_novelity(molcycle_unique_smiles, molcycle_train_smiles + smiles_250k)
    grammar_total_novelty_num += grammar_novelty_num
    molcycle_total_novelty_num += molcycle_novelty_num

print("selected visualization pair:", selected_pair)
print("GrammarVAE result:", selected_grammar_generation)
print("MolCycle GAN result:", selected_molcycle_generation)
print("-----------------------------------------------------")
print("Summary statistics:")
print("-----------------------------------------------------")
total_generated_num = len(success_encode) * (len(success_encode) - 1) / 2 * 56 * 2
print("Total number points tested: {}".format(total_generated_num))
print("Grammar VAE:")
print("Success rate: {:.0%} ({}/{})".format((total_generated_num - grammar_total_failed_number) / float(total_generated_num), total_generated_num - grammar_total_failed_number, total_generated_num))
print("Not identity rate: {:.0%} ({}/{})".format(grammar_total_not_identity / float(total_generated_num), grammar_total_not_identity, total_generated_num))
print("Unique rate: {:.0%} ({}/{})".format(grammar_total_unique / float(total_generated_num), grammar_total_unique, total_generated_num))
print("Novelty rate: {:.0%} ({}/{})".format(grammar_total_novelty_num / float(total_generated_num), grammar_total_novelty_num, total_generated_num))
print("Rule of five rate: {:.0%} ({}/{})".format(grammar_total_rule_of_5 / float(total_generated_num), grammar_total_rule_of_5, total_generated_num))

print("-----------------------------------------------------")
print("MolCycle GAN:")
print("Success rate: {:.0%} ({}/{})".format((total_generated_num - molcycle_total_failed_number) / float(total_generated_num), total_generated_num - molcycle_total_failed_number, total_generated_num))
print("Not identity rate: {:.0%} ({}/{})".format(molcycle_total_not_identity / float(total_generated_num), molcycle_total_not_identity, total_generated_num))
print("Unique rate: {:.0%} ({}/{})".format(molcycle_total_unique / float(total_generated_num), molcycle_total_unique, total_generated_num))
print("Novelty rate: {:.0%} ({}/{})".format(molcycle_total_novelty_num / float(total_generated_num), molcycle_total_novelty_num, total_generated_num))
print("Rule of five rate: {:.0%} ({}/{})".format(molcycle_total_rule_of_5 / float(total_generated_num), molcycle_total_rule_of_5, total_generated_num))
