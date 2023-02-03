
# Explorable Data Consistent CT Reconstruction
This repository is the official implementation of the paper [Explorable Data Consistent CT Reconstruction](https://bmvc2022.mpi-inf.mpg.de/0746.pdf) (British Machine Vision Conference, 2022).

![Here comes the image](./reconstructions.png?raw=true "")

Computed Tomography (CT) is an indispensable tool for the detection and assessment of various medical conditions. This, however, comes at the cost of the health risks entailed in the usage of ionizing X-ray radiation. Using sparse-view CT aims to minimize these risks, as well as to reduce scan times, by capturing fewer X-ray projections, which correspond to fewer projection angles. However, the lack of sufficient projections may introduce significant ambiguity when solving the ill-posed inverse CT reconstruction problem, which may hinder the medical interpretation of the results. We propose a method for resolving these ambiguities, by conditioning image reconstruction on different possible semantic meanings. We demonstrate our method on the task of identifying malignant lung nodules in chest CT. To this end, we exploit a pre-trained malignancy classifier for producing an array of possible reconstructions corresponding to different malignancy levels, rather than outputting a single image corresponding to an arbitrary medical interpretation. The data-consistency of all our method reconstructions then facilitates performing a reliable and informed diagnosis (e.g. by a medical doctor). 

## Get Started

- **Dependencies** 
  - install PyTorch (e.g. version 1.12.1+cu116) and Torchvision (e.g. version0.13.1+cu116) 
  - install requirements by: `pip install -r requirements`
  - install https://github.com/drgHannah/Radon-Transformation
- **Data** 
  - please download the data from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
- **Training and Evaluation** 
	-   for some tests on pretrained networks, save [this network](https://drive.google.com/drive/folders/16pwCuat4tby_O3k2q2JDf79aYd6-cTGb?usp=sharing)  as *network_100a.pt* into the project folder
  - for the optimization, please have a look at  `explorable_reconstruction.ipynb` 


## Specifications used for the publication
- Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz
- GeForce RTX 3090
- CUDA Version: 11.6
