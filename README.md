
# Explorable Data Consistent CT Reconstruction
This repository is the official implementation of Explorable Data Consistent CT Reconstruction, to appear in The 33rd British Machine Vision Conference, 21st - 24th November 2022, London, UK.

![Here comes the image](./reconstructions.png?raw=true "")

## Get Started

- **Dependencies** 
  - install PyTorch (e.g. version 1.12.1+cu116) and Torchvision (e.g. version0.13.1+cu116) 
  - install requirements by: `pip install -r requirements`
- **Data** 
  - please download the data from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
- **Training and Evaluation** 
	-   for some tests on pretrained networks, save [this networks](https://drive.google.com/drive/folders/1WQ_8HSFYS0TAWtHEjCmnh1Tg5avQZcIv?usp=sharing)  as *network_100a.pt* into the project folder
  - for the optimization, please have a look at  `explorable_reconstruction.ipynb` 


## Specifications used for the publication
- Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz
- GeForce RTX 3090
- CUDA Version: 11.6
