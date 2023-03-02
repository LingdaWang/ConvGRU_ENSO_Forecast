# Convolutional GRU Network for the El Nino-Southern Oscillation Region Forecast

## Get Started

### Create an virtual environment using requirements.txt
```
conda install --yes --file requirements.txt
```

### Running details for the CCSM4 dataset:

#### Get the data. This is a preprocessed dataset generously provided by Prof. Dimitrios Giannakis. It is not contained due to size issue, and requests for CCSM4 dataset should be addressed to lingdaw2@illinois.edu


#### Train the ConvGRU network under the dir `ConvGRU_CCSM4`, and the trained models will be saved in `ConvGRU_CCSM4/results/saved_ConvGRU_model`
```
CUDA_VISIBLE_DEVICES=0 python main.py
```

#### Switch to the dir `ConvGRU_CCSM4/results`, and test the trained model
```
CUDA_VISIBLE_DEVICES=0 python ConvGRU.py
```

#### Get results for other methods
```
python GRU.py
python LIM.py
python LR.py
```

#### Switch to the dir `ConvGRU_CCSM4/results/plots` and use plot_res.ipynb for figures


