# *Fair Text Classification with Wasserstein Independence*

This repository allows to run the code to reproduce the 
experiments of the paper *Fair Text Classification with Wasserstein Independence*.
The original code comes from the Fairlib library (https://github.com/HanXudong/fairlib), 
a framework in which we have integrated our approach.

## Data
To download the dataset used in our experiments, please use the following commands:
- Bias in Bios dataset : ***python3 download_data.py Bios***
- Moji dataset : ***python3 download_data.py Moji***
- Marked Personas dataset : ***python3 download_data.py Marked_personas***  

*Note:* for the EEC dataset, no download is required, the dataset will
be temporarely downloaded when training the demonic model.

## Training *demonic* model
To train the *demonic* model, please use the following commands:
- for the Bios dataset :
  - without transfer :  
  **python3 train_demonic/train_demonic_Bios.py Bios**
  - demonic pretrained on the EEC dataset :  
    **python3 train_demonic/train_demonic_Bios.py EEC**
  - demonic pretrained on the Marked Personas dataset :  
    **python3 train_demonic/train_demonic_Bios.py dv2_story**
- for the Moji dataset :  
**python3 train_demonic/train_demonic_Moji.py Moji**

## Running experiments

To launch our **default models**, please run the following commands:
- for the Bios dataset: ***python3 main.py Bios Bios***
- for the Moji dataset: ***python3 main.py Moji Moji***

To launch our models on the Bios dataset, using a *demonic* model 
pretrained on other domains, please use:
- for the *demonic model* pretrained on the EEC dataset:  
***python3 main.py Bios EEC***
- for the *demonic model* pretrained on the Marked Personas dataset:  
***python3 main.py Bios marked_personas***
