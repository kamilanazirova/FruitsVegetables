# FruitsVegetables
The project is a telegram bot for recognizing fruits and vegetables.

![image](https://domf5oio6qrcr.cloudfront.net/medialibrary/11499/3b360279-8b43-40f3-9b11-604749128187.jpg)

# MVP1
- Initializing the project
    - Setting up the environment and installing the necessary librarie
    - Creating a repository to store code and data

- Data collection 
    - Identifying data sources
    - Extracting data from sources
    - Pre-processing of data (cleaning, transformation, aggregation, etc.)

- Data processing
    - Data exploration and visualization
    - Data transformation and cleaning (handling missing values, scaling, categorical feature coding, etc.)

- Machine learning
    - Model selection (regression, classification, clustering, etc.)
    - Dividing the data into training and test sets
    - Training the model on the training set
    - Model evaluation on a test set (performance evaluation, cross-validation, etc.)
    - Fine-tuning model hyperparameters

- ClearML Integration
    - Creating a project in ClearML
    - Code integration with ClearML to automatically track experiments and metrics 
    - Load data and models into ClearML for further analysis and visualization of results.

- Interface development using Streamlit
    - Creating an interface for interacting with the trained model and data
    - Adding controls (sliders, buttons, text fields, etc.) for entering user data
    - Visualization of the results of the model (graphs, tables, etc.)

 ### Project participants: 
 Liliya Vasilova, 
 Lyaisan Mukhanedyanova, Kamila Nazirova, Ulyana Khamidullina

# List of libraries used
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from clearml import Task, Logger
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import os
    import time
    import rembg
    from efficientnet_pytorch import EfficientNet
    import torchvision.models as models
    import torch.nn as nn
    from torchmetrics import Precision, Recall, F1Score
    import torch.optim.lr_scheduler as lr_scheduler

# Links
[Dataset](https://github.com/kamilanazirova/Dataset_FruitsVegetables)
[ClearMl](https://app.clear.ml/projects/523ccba1903b47fa915ebefc7b7e3fd6/experiments/e7fb800483704c8fb94b8da31e234df9/output/execution)
[Dataset](https://www.kaggle.com/datasets/moltean/fruits)