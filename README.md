# autoqtl

[![Logo](./docs/FinalLogo_Big.png)]()

AutoQTL : Automated Quantitative Trait Locus Analysis
==================================

**AutoQTL** is an automated machine learning tool for QTL analysis.
The goal of AutoQTL is to automate QTL analysis by building an analytics pipeline optimized for explaining variation in a quantitative trait given a set of genetic variants. It uses genetic programming (GP) as the search and optimization method. 

AutoQTL is recommended to be used as a posthoc analysis to genome-wide association/QTL analysis. AutoQTL aims to provide additional insights into the association of phenotype to genotype including, but not limited to, the detection of non-additive genetic inheritance models and epistatic interactions. Furthermore, our feature importance metrics, in tandem with summary statistics, can provide additional evidence for the identification of putative QTL and targets for gene set enrichment and KEGG pathway analysis. 

#geneticsmeetsautoML

## Installing & Running AutoQTL

We recommend installing the Python package 'autoqtl' from the pypi repository to run AutoQTL. 
[The package can be found at] (https://pypi.org/project/autoqtl/#description) 

We also recommend using conda environments for installing Autoqtl, but it is not necessary. 
Recommended installation:
```
conda create --name autoqtl_env python=3.10
conda activate autoqtl_env
pip install autoqtl
```

After installation, the 'demo.ipynb' jupyter notebook in the tutorials folder can be used as a reference to run AutoQTL.  

Anyone interested in exploring the code base of AutoQTL further, can clone the repository, make a conda environment using the requirements.txt file and try out new things.

This software is built as part of a proof-of-concept and hence is still under development.  
We continue to work on to add new features and functionality to AutoQTL. 
Suggestions are welcome.

## License

Please see the [repository license](https://github.com/EpistasisLab/autoqtl/blob/master/LICENSE) for licensing and usage information.
Autoqtl is open source and freely available but citation is required.

## Citing AutoQTL

If you use AutoQTL in a scientific publication, please consider citing the following paper:

Philip J. Freda, Attri Ghosh, Elizabeth Zhang, Tianhao Luo, Apurva S. Chitre, Oksana Polesskaya, Celine L. St. Pierre, Jianjun Gao,
Connor D. Martin, Hao Chen, Angel G. Garcia-Martinez, Tengfei Wang, Wenyan Han, Keita Ishiwari, Paul Meyer, Alexander Lamparelli,
Christopher P. King, Abraham A. Palmer, Ruowang Li and Jason H. Moore. [Automated quantitative trait locus analysis (AutoQTL)](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-023-00331-3). *BioData Mining* 16, Article number: 14 (2023)

