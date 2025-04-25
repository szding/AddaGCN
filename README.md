# AddaGCN
AddaGCN: spatial transcriptomics deconvolution using graph convolutional networks with adversarial discriminative domain adaptation

## Overview
Schematic view of the AddaGCN framework.
![AddaGCN_Overview](https://github.com/szding/AddaGCN/blob/main/AddaGCN_Overview.png)

**a**, First, we employs the scRNA-seq reference data to identify cell-type marker genes and generate a pseudo-spot pool. Than, the pseudo-ST and real-ST spots are constructed as a combined graph by their transcriptional similarity (MNN) and Spatial Nearest Neighbors (SNN). Finally, AddaGCN is used to output the cell-type composition for real-ST spots. b, The model training process involves initialization followed by two optimization steps. The initialization step focuses on training the source classifier model to minimize the $L_s$ loss function, enabling the prediction of cell type proportions from pseudo-spots. The two subsequent optimization steps transition the process into adversarial domain adaptation. In the first optimization step, the network is trained to minimize $L_{total}$, with the weights of the domain classifier $D(\cdot )$ fixed. In the second optimization step, the domain labels are first reversed, and than the domain classifier is trained to minimize $L_{adv,2}$, while the weights of the GCN feature extractor $f(\cdot )$ and source classifier $S(\cdot )$ remain fixed. These two steps are alternated over a predetermined number of iterations. Finally, the trained model, AddaGCN, is capable of predicting cell type proportions for real ST data.


## Installation
The AddaGCN package is developed based on the Python libraries [Scanpy](https://scanpy.readthedocs.io/en/stable/), [TensorFlow](https://www.tensorflow.org/) and [Keras](https://github.com/keras-team/keras) framework, and can be run on GPU (recommend) or CPU.

First clone the repository. 

```
git clone https://github.com/szding/AddaGCN.git
cd AddaGCN-main
```

It's recommended to create a separate conda environment for running AddaGCN:

```
#create an environment called AddaGCN
conda create -n AddaGCN python=3.8

#activate your environment
conda activate AddaGCN
```
Install all the required packages. 

For Linux
```
pip install -r requirement.txt
```
Install AddaGCN.

```
python setup.py build
python setup.py install
```

## Tutorials

Three step-by-step tutorials are included in the `Tutorial` folder to show how to use AddaGCN.
- Tutorial_Developing_human_heart: This exploration is based on three developmental stages of the human embryonic heart, i.e., 4.5-5 PCW, 6.5 PCW, and 9 PCW.
- Tutorial_mouse_brain: The anterior section of mouse brain.
- Tutorial_PDAC-A: The pancreatic ductal adenocarcinoma (PDAC) slice A.
- Tutorial_PDAC-B: The pancreatic ductal adenocarcinoma (PDAC) slice B.
- Tutorial_seqFISH+: A simulated spatial transcriptomic dataset (seqFISH+ dataset).
  
## data
Here is the demo data in this article for the jupyter file [PDAC-A](https://github.com/szding/AddaGCN/blob/main/Tutorials/Tutorial_PDAC-A.ipynb).

## Support
If you have any questions, please feel free to contact us dszspur@xju.edu.cn. 











