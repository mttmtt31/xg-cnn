# xG-CNN
This repository allows you to fully reproduce the results of our research we submitted to the 10th Workshop on Machine Learning and Data Mining for Sports Analytics (Turin, 2023).

You can train the model yourself either re-creating the input from scratch, or using a provided input. The repository also includes a notebook called `Demo XG-CNN` which loads a pre-trained model and computes the xG for every shot taken by Lionel Messi in his last La Liga season in Barcelona.

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/mttmtt31/xg-cnn.git

2. Install the packages
   ```shell
   cd xg-cnn
   pip install -r requirements.txt

# Download JSON File from Google Drive

To access the JSON file used to train the model, you can either manually download it into the `data` folder from [this Google Drive link](https://drive.google.com/file/d/17FzPzXPstohNNHc-qdeCUKXPOQVqFdwy/view?usp=drive_link), or you can do it automatically following the instructions below:

1. **Make the download script (bash) executable** with the following command:

   ```bash
   chmod +x download.sh

3. **Run the script** using the following command
   ```bash
   bash download.sh

If you download it manually, make sure the JSON file is stored as `data/shots.json`.

Alternatively, the JSON file can be recreated from scratch (with the most updated version of the StatsBomb open data), by running the following command:

   ```shell
   python src/utils/create_json.py
   ```

Please note that this piece of code was not optimised, as it was meant to be run only once.

# Prepare the dataset
Once the JSON is downloaded, it needs to be turned into a labeled dataset. Each element of the dataset is a numpy array of shape $2\times30\times40$. The label is $1$ if the shot was converted into a goal, or 0. 

To create the labeled dataset, run:
   ```shell
   python src/utils/create_heatmaps.py
   ```


# Train the model
To train the model, run the `main.py` script:
```shell
python main.py --data --labels --device --batch-size --learning-rate --dropout --epochs --augmentation --wandb --save --optim --weight-decay --gaussian-filter
```
The user can play with the parameters of the model (like the optimiser **--optim**, or the learning rate **--learning-rate**), and with the Gaussian smoothing filter to apply to each array (**--gaussian-filter**). Additionally, the user can specify some boolean parameters, like **--augmentation** to double the data size as described in the paper, **--save** to save the PyTorch model in the `trained_models` folder, and **--wandb** to automatically log results in wandb (if specified, please make sure wandb is currectly configured on your end) in a new project called `xg-cnn`.

For the complete documentation, please run: 

```shell
python main.py --help
```

# Demo
The `Demo XG-CNN.ipynb` notebook allows the user to explore the results obtained with our pretrained model, by computing the xG for each shot taken by Lionel Messi during his last season in Barcelona.
