# PlanktID
Plankton identification and classification for the [National Data Science Bowl](https://www.kaggle.com/c/datasciencebowl) Kaggle challenge.

This is first and foremost an attempt at the challenge but also a way of generalizing Deep Learning
experimentation with PyTorch.

## Configuration
The `options.json` file controls all configuration options for training and evaluation

## Data
The basic assumption for this setup is that the data for training, validation, and testing is split into
folders representing the classes. For example:
```
-- cats
    -- cat1.jpg
    -- cat2.jpg
    -- ...
-- zebras
    -- zebra1.png
    -- zebra_safari.jpg
    -- ...
```

As such the naming scheme of the actual training files is less important than the directory structure.
Upon initial training of the model the dataset will be split into the `train/val/test` partitions
and `csv` files will be created for each class indicating which files will be taken for training.


## Training
`main.py` can be run to train the model for classification. 