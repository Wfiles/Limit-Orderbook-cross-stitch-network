# CS-433 Machine Learning - Project 2 Report - Multi-Horizon Volatility Forecasting using Ordinary Differential Equation and Cross-Stitch Networks

The aim of this project is to introduce a novel architecture for predicting the volatility of Limit Order Books using the FI-2020 Dataset. Our approach combines the strengths of Ordinary Differential Equation (ODE) networks and Cross-Stitch networks to address the challenges of multi-horizon forecasting in high-frequency trading environments. To evaluate the effectiveness of our approach, we compare it against the Temporal Fusion Transformers (TFT) architecture, a state-of-the-art method for temporal forecasting tasks.

Our group is composed of : 
- William Jallot 341540
- Thierry Sokhn 345880
- Matthias Wyss 329884

## Installation
```
conda create --name <env_name> python=3.11.9
conda activate <env_name>
pip install -r requirements.txt
```
The requirements.txt has been generated from an Apple Intel computer.

The `run.ipynb` file contains the final run code for the two models we obtained.

The data should be inside the `./data` folder (if it does not exist create it, using the following link: https://drive.google.com/file/d/1OFdK_Ya_wJEgCKqAaF5RxyR9V8yOX0IS/view?usp=drive_link). This data has been fetched and selected from https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649 which are the first publicly available datasets that contain representations and annotations for a limit order book in the High Frequency universe. 

#### Implemented Models:
- `Cross-Stitch Network using Ordinary Differential Equations`
- `Temporal Fusion Transformer`

All the models are being run inside the `run.ipynb` file.

#### Project structure

```
.
│   .gitignore
│   run.ipynb
│   README.md
│   requirements.txt
│
├───checkpoint
├───data
│       Test_Dst_NoAuction_DecPre_CF_7.txt
│       Test_Dst_NoAuction_DecPre_CF_8.txt
│       Test_Dst_NoAuction_DecPre_CF_9.txt
│       Train_Dst_NoAuction_DecPre_CF_7.txt
│
├───Cross_Stitch_Models
│   │   cross_stitch_network.py
│   │   inception.py
│   │   linear_cross_stitch_unit.py
│   │   loop_layer.py
│   │   odefunc.py
│   │   ode_layer.py
│   │   reshape.py
│
├───Temporal_Fusion_Transform
│   │   README.md
│   │   tft_model.py
│   │
│   ├───data_formatters
│   │   │   base.py
│   │   │   ts_dataset.py
│   │   │   utils2.py
│   │   │   volatility.py
│   │   │   __init__.py
│   │
│   └───expt_settings
│       │   configs.py
│       │   __init__.py
│
└───utils
    │   dynamic_losses.py
```

The project structure includes a folder `Cross_Stitch_Models` containing our Cross-Stitch ODE network with the different layers, while our folder `Temporal Fusion Transform` implements our temporal fusion transformer.


#### Useful functions implemented
- What was handled during the data cleaning / feature processing:
  - Midprice computation
  - Stocks splitting 
  - Log returns computation of the volatility
  - Standardization (z-score).
  - Predictions upscaling (reverted when computing performance metrics)

- What has been used for evaluation metrics
  -  Mean Relative Absolute Error