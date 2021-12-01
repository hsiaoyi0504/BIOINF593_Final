# BIOINF 593 Final

## Create environment on Greatlakes
``` shell
module load python2.7-anaconda/2019.03
conda env create -f environment.yml -n BIOINF593
```

## Test interporation
``` shell
cd grammarVAE
python grammarVAE_interpolation.py
```

## Reference
- GrammarVAE: [Repo](https://github.com/mkusner/grammarVAE)
- Mol-CycleGAN: [Repo](https://github.com/ardigen/mol-cycle-gan)
