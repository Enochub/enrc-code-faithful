# Deep Embedded Non-Redundant Clustering (ENRC)

Implementation of [Deep Embedded Non-Redundant Clustering (ENRC)](https://ojs.aaai.org/index.php/AAAI/article/view/5961) accepted at AAAI 2020.

The supplementary pdf file can be found [here](supplementary.pdf) and the used data sets have been uploaded at figshare [here](https://figshare.com/articles/ENRC/12272921).


# Code

#### Update (July, 2023): We released [ClustPy](https://github.com/collinleiber/ClustPy), an open source (deep) clustering python package that offers an easy to use implementation of ENRC.

The code of ENRC has been implemented in Python 3.6 and tested on Ubuntu 18.04.2 LTS.
We implemented the autoencoder and ENRC algorithm in Pytorch 1.1.0 with cuda support and used for the pretraining several methods from fastai 1.0.54.

Other used packages:
numpy, pandas, scikit-learn, scipy, torchvision, pillow

**Note:** If you want to use the most recent version of pytorch you have to update the fastai version to its latest version as well. 
Unfortunately, fastai introduced some breaking changes in its interface, so you might have to update parts of the code (e.g. imports, function names, etc.).



# Reproducing the experiments

After installing the required packages you can run the experiments by calling the separate cluster_{dataset_name}.py files, e.g.:
> python cluster_stickfigures.py

This will automatically pretrain ten autoencoders and run ENRC afterwards for the stickfigures data set.

# Errata

Equation 2 of the main paper is missing a power of two, it should be: 
```math
|| a - b||_{\tau}^2:=\sum_{i=1}^{D}\tau[i]^2(a[i]-b[i])^2
```

# Acknowledgements

We thank Collin Leiber (LMU Munich) for his python implementation of NR-Kmeans, which we adapted for our purposes.

# Citation (AAAI)
```
@article{Miklautz_Mautz_Altinigneli_Böhm_Plant_2020,
 title={Deep Embedded Non-Redundant Clustering},
 volume={34},
 url={https://ojs.aaai.org/index.php/AAAI/article/view/5961},
 DOI={10.1609/aaai.v34i04.5961}, number={04},
 journal={Proceedings of the AAAI Conference on Artificial Intelligence},
 author={Miklautz, Lukas and
         Mautz, Dominik and
         Altinigneli, Muzaffer Can and
         Böhm, Christian and
         Plant, Claudia},
         year={2020},
         month={Apr.},
         pages={5174-5181} 
}
```

