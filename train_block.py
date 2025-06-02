from data import * 
from model import *
data = Dataset()
data.to_torch()
data.train_val_test_split()
model = AttentionModel()
model.train_model(data=data)