from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors


def get_rule_of_five_violation_num(smiles):
    vio_num = 0
    m = Chem.MolFromSmiles(smiles)
    mw = Descriptors.ExactMolWt(m)
    if mw >= 500:
        vio_num += 1
    h_bond_donor = Descriptors.NumHDonors(m)
    if h_bond_donor > 5:
        vio_num += 1 
    h_bond_acceptors = Descriptors.NumHAcceptors(m)
    if h_bond_acceptors > 10:
        vio_num += 1
    logP = Descriptors.MolLogP(m)
    if logP >= 5:
        vio_num += 1
    print(smiles, mw, vio_num)

selected_pair =  ['CC(=O)O[C@H]1CC[C@]2(C)C3CC[C@@]4(C)C(CC=C4C4=CN=CC=C4)C3CC=C2C1', 'CC(=O)NCCCS(O)(=O)=O']
print("selected pair:")
for s in selected_pair:
    get_rule_of_five_violation_num(s)
print("---------------------------------------------------------------")

GrammarVAE_result = ['NC(=O)NCCOOC(=O)[O-]', 'CC(=O)CNCCOC(=O)', 'CC(=O)NCCC/NC(=O)', 'CC(=O)NCCCOC(=O)', 'CC(=O)NCCCN/C(=O)', 'CC(=O)NCCOOC(=O)', 'CC(=O)NCC#CPC(=O)', 'CC(=O)NCCCPC(=O)', 'CC(=O)NCCCCC(=O)', 'CC(=O)NCC#C/P(=S)=O', 'NC(=O)NCCCNC(=O)O', 'CC(=O)NCOCCC(=O)', 'CC(=O)NCC=CNC(=O)', 'CC(=O)NCC=NP(=S)=O', 'CC(=O)NCCCNC(=O)', 'CC(=O)NSCCOC(=O)', 'CC(=O)NN=CC/C(=O)[O-]', 'CC(=O)NNCC#P/C(=O)', 'CC(=O)NCOCOC(=O)', 'CC(=O)NCCC=P(\\S)=O', 'CC(=O)NNCNPC(=O)', 'NC(=O)CCCOOC(=O)[O-]', 'CC(=O)NNCCOC(=O)', 'CC(=O)NNCCPC(=O)', 'CC(=O)NCCC#PC(=O)', 'CC(=O)NCCCOP(=O)', 'CC(=O)NCOCNC(=O)', 'CC(=O)NCSCPC(=O)', 'CC(=O)NCSCNC(=O)']

print("GrammarVAE:")
for s in GrammarVAE_result:
    get_rule_of_five_violation_num(s)
print("---------------------------------------------------------------")

MolCycle_GAN_result = ['CC(=O)O[C@H]1CC[C@H]2[C@H](C)c3nc(C4CCCC4)sc3C[C@H]2[C@H]1C', 'CC(=O)O[C@H]1C=C[C@@H]2[C@@]3(C)CC[C@H](C)C[C@@H]3CC[C@@]12C', 'CC(=O)C1=CC[C@@H](Oc2nc3c(s2)CC[C@@H](C)[C@H]3C)C12CCCCC2', 'C[C@@H]1NN=C[C@H]1CNS(=O)(=O)CCc1ccco1', 'CC(=O)O[C@H]1[C@H]2[C@H]([C@H]3CCCO3)CCC[C@]2(C)C2=C3N=C(C)S[C@@H]3CC[C@H]21', 'CC(=O)O[C@H]1C2=C3N=C(C)S[C@H]3CC[C@H]2[C@]2(C)CCC[C@H]([C@H]3CCCO3)[C@H]12', 'CC(=O)O[C@@H]1CC[C@]2(C)C(=CC[C@H]3[C@@H]4CC=C([C@H]5CCCO5)[C@@]4(C)CC[C@H]32)C1', 'CC(=O)C1=Cc2csc(C)c2[C@H]1O[C@H]1CO[C@H]2CCCC[C@H]2C1(C)C', 'CC(=O)NCCCS(=O)(=O)[O-]', 'CC(=O)/N=C\\CCS(C)(=O)=O', 'Cc1cc(C)c2c(n1)C1(CCC1)[C@@H](C(=O)OCC(=O)[O-])CC2', 'CC(=O)O[C@@H]1C[C@H]([C@H]2CC[C@H]3CCCC=C3C2)O[C@H]2CCC[C@H](N)[C@@]21C', 'CC(=O)O[C@@H]1C2=C3N=C(C)S[C@H]3CC[C@@H]2[C@@]2(C)CCC[C@@H]([C@@H]3CCCO3)[C@@H]12', 'CC(=O)O[C@H]1C[C@H]([C@@H]2CCCC3=CCCC[C@@H]32)O[C@H]2CC[C@]3(CO3)[C@@]12C', 'CC1=CCC2(CCC3(CC2)O[C@H](C)[C@@]2(O)[C@@H](C(=O)[O-])CO[C@]32C)NC1', 'CC(=O)O[C@H]1CC[C@]2(C)CC[C@H]3CC[C@]4(C)C(=CC[C@H]4Br)C3=C2C1', 'CC(=O)O[C@H]1C2=C3N=C(C)S[C@@H]3CC[C@@]2(C)[C@H]2CC(=O)O[C@@H]12', 'CC(=O)O[C@@H]1CC[C@H]2[C@H](C)c3nc([C@H]4CCCCC4=O)sc3C[C@]21C', 'CC(=O)O[C@H]1CC[C@H]2[C@@H](C)c3nc([C@@H]4CCCC[C@H]4Br)sc3C[C@@]12C', 'CC(=O)NCC[C@@H](C)[S@](C)=O', 'C[C@]1(CNS(=O)(=O)CCCO)CN(N)CS1', 'CC(=O)O[C@H]1CC[C@@H]2CCO[C@@]3(OCC[C@H]3C(N)=O)[C@]2(C)[C@@H]1C', 'CC(=O)O[C@H]1CC[C@@H]2CC=C3C[C@@H](c4ccncc4)CC[C@]3(C)[C@H]2[C@@H]1C', 'CC(=O)Oc1cnc2c(c1C)[C@@H]([C@H]1CCCO1)C1=C(C2)C[C@H](C)CC1', 'CC(=O)Oc1ncc2c(c1C)[C@H]1C(=CC2)CC[C@@H]([C@@H]2CCCO2)[C@@H]1CO', 'CC(=O)O[C@H]1[C@H]2Cc3sc(C)nc3C[C@H]2[C@@]2(C)CCC[C@H]([C@H]3CCCO3)[C@H]12', 'CC(=O)Oc1cc2c(nc1C)CC=C1C[C@@H]([C@H]3CCCO3)CC[C@]12C', 'C[C@@H](C#N)OCCCN1C=NC[C@@H](C)C1=O', 'CC(=O)N1C[C@@H](O)[C@@H]2[C@H](c3cc(C)cs3)[C@@H](C3CCCC3)CC[C@H]21', 'CC(=O)O[C@H]1C[C@H]([C@@H]2CCCC3=CCCC[C@@H]32)O[C@H]2C(O)=CC[C@@]12C', 'CC(C)(O)/N=C/CCCN1CNc2ccccc21', 'CC(=N)CS(=O)(=O)CCCNC(C)=O', 'CC(=O)O[C@@H]1CC[C@H]2CC=C3C[C@H]([C@H]4CCCO4)CC[C@]3(C)[C@H]2[C@H]1C', 'CC(=O)NCCC[S@](=O)O', 'CN1CNCN(CC(=O)CCO)C1', 'Cc1cc(C)c(NCCOC2(C)CCCCC2)s1', 'CC1(C)[C@@H](CONC(=O)[C@H]2CCCO2)CO[C@@H]2CCCC[C@@H]21', 'CC(=O)CNS(C)(=O)=O', 'CC(=O)CNS(=O)(=O)CCCO', 'Cc1ccc(N2C(=O)O[C@@H]3[C@H](C#N)[C@@H]4CC[C@@H](C)[C@H]4[C@H](C)[C@H]32)o1', 'CC(=O)C1=CCC[C@@H]1O[C@@H](C)[C@H]1CCC[C@H]2SC3=C(CC=CC3)[C@@]12C', 'C[C@H]1C(=O)OC2(CCCCC2)[C@@]1(N)[C@H]1CC[C@@H]2CCC[C@@H]2C1', 'CCOC(=O)CCNC(C)=O', 'CC1(C)C(=O)O[C@@]2(C)C1=CC=C1OCC(c3nc4c(s3)CCCC4)=C12', 'CC(C)/N=C/CCNS(=O)(=O)Cc1ccon1', 'CC(=O)O[C@H]1[C@@H](C)[C@@H]([C@@H]2CCCC3=CCCC[C@H]32)OCC1(C)C', 'CC(=O)O[C@H]1CC(=O)[C@@H](C)[C@@]1(C)[C@@H]1CCOC2(CCCCC2)C1', 'CC(C)/N=C/C[C@H]1C=C(CCl)C=[NH+]1', 'CC(=O)NCCCC[S@@](=O)O', 'CC(=O)O[C@H]1CCC2=CC[C@H]3OCC[C@H](C4(Br)CCC4)[C@@H]3[C@]2(C)[C@H]1C', 'CC(=O)O[C@H]1C[C@H]([C@@H]2CCCC3=CCCC[C@H]32)O[C@]2(O)COCC[C@@]12C', 'CC1=C2[C@@H](CC1)CC[C@H]1[C@H](C)[C@@H](OC(=O)c3ccncc3)CC[C@@H]21', 'C[C@H]1CN(CCCS(=O)(=O)CO)C(=S)S1', 'CC(=O)NCCC[SH](=O)(O)C[NH3+]', 'CC1=NC2=C3[C@@H](CC[C@@H]2S1)[C@@H](OC(=O)[C@H]1C[C@@H]2CC[C@H]1O2)[C@H]1[C@H](C)CCC[C@@]31C', 'CC(=O)O[C@H]1C[C@@H]([C@@H]2CCCC3=CCCC[C@@H]32)OC2=Cc3ncccc3[C@]21C', 'CC(=O)O[C@@H]1CC[C@H]2CC=C3[C@@H](CC[C@H]4CC[C@H]5CC=C[C@H]5[C@]34C)[C@@]21C', 'C[C@H]1[C@@H](C)[C@@H](OC(=O)[C@H]2OCOC2(C)C)CO[C@]12CCCCC21CC=CC1', 'CC1(C)[C@@H]2CC=C(CBr)[C@@]2(C)CC[C@@]1(C)O', 'CC(C)/N=C/CCNS(C)(=O)=O', 'Cc1cc(C)c(N2C(=O)O[C@@H]3CO[C@H]4C[C@@H]5CCCC[C@]5(C)[C@@]4(C#N)[C@H]32)o1', 'CC(=O)O[C@H]1Cc2sc(C3CCC4(CCCO4)CC3)nc2[C@@H](C)[C@H]1C', 'CC(=O)C/N=C/C[C@@H](O)c1nc2ccccc2[nH]1', 'CC(=O)NCCCS(C)(=O)=O', 'CC(=O)C1=C(C)[C@@H](O[C@H]2CCCOC23CCCCC3)[C@@]2(CCc3c(C)csc3C2)C1', 'CC[SH](=O)(O)CCCNC(C)=O', 'CC(=O)NCCC[S@@](=O)CC#N', 'CC(=O)O[C@H]1CCc2nc([C@H]3OCC(=O)C34CCCCC4)sc2[C@@H]1C', 'CC(C)=NC(C)(C)S(=O)(=O)N1CCSC1', 'CC(=O)O[C@H]1[C@H]2CC[C@@H]3SC(C)=NC3=C2[C@]2(C)CCC[C@@H](CO)[C@@H]12', 'C[C@@H]1C[C@H](Br)CC[C@H]1[C@H]1[C@@](C)(O)C[C@]1(C)C1=CC=CCC1=O', 'CC(=O)O[C@H]1C[C@H]([C@@H]2CCCC3=CCCC[C@H]32)O[C@]2(C)CCC[C@@]12O', 'C[C@H](C[C@H]1CC[C@H]2CCCC=C21)[NH2+]C1CCCC1', 'CC(=O)/N=C\\CCO', 'CC(=O)NCCCCS(=O)(=O)O', 'CC1=C(CN(C)CCCOc2ccccc2)CS(=O)(=O)C1', 'COCCCS(=O)(=O)NCC(C)(C)C1=c2ccccc2=[NH+]C1', 'CC(=O)O[C@H]1CC=C2[C@@]3(C)Cc4nc(C)sc4C[C@@H]3CC[C@]21C', 'CC(=O)NCCC[S@@](=O)O']

print("MolCycle GAN:")
for s in MolCycle_GAN_result:
    get_rule_of_five_violation_num(s)


