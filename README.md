# Convolutional GRU Network for the El Nino-Southern Oscillation Region Forecast

## 1. Create an virtual environment using requirements.txt
```
conda install --yes --file requirements.txt
```

## 2. Running details for the CCSM4 dataset:

### 2.1 Get the data. This is a preprocessed dataset generously provided by Prof. Dimitrios Giannakis. It is not contained due to size issue, and requests for CCSM4 dataset should be addressed to lingdaw2@illinois.edu


### 2.2 Train the ConvGRU network under the dir `ConvGRU_CCSM4`, and the trained models will be saved in `ConvGRU_CCSM4/results/saved_ConvGRU_model`
```
CUDA_VISIBLE_DEVICES=0 python main.py
```

### 2.3 Switch to the dir `ConvGRU_CCSM4/results`, and test the trained model
```
CUDA_VISIBLE_DEVICES=0 python ConvGRU.py
```

### 2.4 Get results for other methods
```
python GRU.py
python LIM.py
python LR.py
```

### 2.5 Switch to the dir `ConvGRU_CCSM4/results/plots` and use `plot_res.ipynb` for figures


## 3. Running details for the NOAA-GDFL-SPEAR dataset:

### 3.1 Download the data to the dir `ConvGRU_SPEAR/data`, and instructions can be found at `ConvGRU_SPEAR/data/README.md`

### 3.2 Train the ConvGRU network under the dir `ConvGRU_SPEAR`, and the trained models will be saved in `ConvGRU_CCSM4/results/saved_ConvGRU_model`
```
CUDA_VISIBLE_DEVICES=0 python main.py
```

### 3.3 Switch to the dir `ConvGRU_SPEAR/results`, and test the trained model
```
CUDA_VISIBLE_DEVICES=0 python ConvGRU.py
```

### 3.4 Get results for other methods
```
python GRU.py
python LIM.py
python LR.py
```

### 3.5 Use `plot_res.ipynb` for figures



