import sys
import torch
from torchmetrics import AUROC
from torchmetrics import ROC
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryF1Score
import argparse
import ROOT
import pandas as pd
import numpy as np
import pathlib
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score
ROOT.gStyle.SetOptStat(0)

def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)


class EarlyStopping:
    def __init__(self, patience=100, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), f"../PNN_models/checkpoint_m:{SUSY_model}_lep:{lepton}_lr:{lr}_wd:{wd}_.pt")
        self.val_loss_min = val_loss
        
class PNN(nn.Module):
    def __init__(self, num_features, neuron, p=0.2):
        super(PNN, self).__init__()
        
        self.fc1 = nn.Linear(num_features, neuron)
        self.bn1 = nn.BatchNorm1d(neuron)
        self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(p=p)
        
        self.fc2 = nn.Linear(neuron, neuron)
        self.bn2 = nn.BatchNorm1d(neuron)
        self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(p=p)
        
        self.fc3 = nn.Linear(neuron, neuron)
        self.bn3 = nn.BatchNorm1d(neuron)
        self.relu3 = nn.ReLU()
        # self.dropout3 = nn.Dropout(p=p)
        
        self.fc10 = nn.Linear(neuron, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        # x = self.dropout3(x)
        
        x = self.fc10(x)
        return x

# define input parameters for the script
parser = argparse.ArgumentParser(description="Process some training parameters.")
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--chunks', type=int, default=1, help='Number of chunks')
parser.add_argument('--blocks', type=int, default=64, help='Blocks per chunk')
parser.add_argument('--model', type=str, choices=['C1C1', 'C1pN2', 'C1mN2'], help='Type of model')
parser.add_argument('--lepton', type=str, choices=['el', 'mu'], help='Lepton type to be skimmed')
parser.add_argument('--name', type=str,  help='Name of input file')
parser.add_argument('--imb', type=float, default=50, help='Imbalance factor to scale down class weight')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='Weight decay')
parser.add_argument('--type', type=str, choices=['root', 'csv', 'parquet'], default='ttree', help='File type')
args = parser.parse_args()

epochs = args.epochs
number_of_chunks = args.chunks
blocks_per_chunk = args.blocks
lepton = args.lepton
model = args.model
name = args.name
lr = args.lr
wd = args.wd
Type = args.type
Type = 'parquet'

imb = args.imb
print(f"epoch = {args}")
print(f"number_of_chunks = {args}")
print(f"blocks_per_chunk = {args}")

# 
event_info = ["EventNumber", "RunNumber", "dsid", "model"]
particle_info = ["lep_id", "leading_lep"]
kinematics = ["lep1_pt", "lep2_pt", "lep1_eta", "lep2_eta", "lep1_phi", "lep2_phi", "mll", "mt", "mt2_core", "ev_njet", "met_core_sumet"]
jets = ["jet1_pt", "jet1_eta", "jet1_phi", "jet_sumpt"]
model_parameters = ["m1", "m2"]

# define dataset features
training_label = ["Label"]
training_features = kinematics + model_parameters + training_label

SUSY_model = model
mll_cut = 15

# load dataset
input_file = f"../data/SM_SUSY_{SUSY_model}_{lepton}_mll-{mll_cut}_{name}"
df = ROOT.RDataFrame("CollectionTree", f"{input_file}.root")

# training configuration
downsample = False
dataset_size = df.Count().GetValue()
chunk_size = int(dataset_size / number_of_chunks)
block_size = int(chunk_size / blocks_per_chunk)

print("Events in dataset ", dataset_size)
if downsample:
    Batch_size = 14000
    batch_size = 1000

elif not downsample:
    Batch_size = 1000

batches_in_memory = int(dataset_size / Batch_size)

SM_size = df.Filter('Label == 0').Count().GetValue()
SUSY_size = df.Filter('Label == 1').Count().GetValue()

dsid_dfs = []
for file in [f for f in pathlib.Path("../data/dsid_files").iterdir() if f.is_file()]:
    dsid_dfs.append(ROOT.RDataFrame("CollectionTree", str(file)))

print("size of dsid_dfs ", len(dsid_dfs))

print(SUSY_size)
print(f"SUSY fraction in dataset: {SUSY_size / dataset_size}")

num_features = len(training_features) - 1
print(f"features: {num_features} ")

if Type == "csv":
    validation_split = 0.30
    batch_size = 1000
    pdf = pd.read_csv(f"{input_file}.csv")
    all_columns = training_features
    all_columns.remove("Label")
    input_np_array = pdf[all_columns].to_numpy()
    target_np_array = pdf[["Label"]].to_numpy()
    inputs = torch.tensor(input_np_array, dtype=torch.float32)
    targets = torch.tensor(target_np_array, dtype=torch.float32)
        
    ds = TensorDataset(inputs, targets)
    train_dataset, val_dataset = torch.utils.data.random_split(ds, [1 - validation_split, validation_split])
    
    print("####### PyTorch DataLoader (CSV) ###############")
    gen_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    gen_validation = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

elif Type == "parquet":
    print("####### Hugging Face DataLoader (Parquet) ###############")
    from datasets import load_dataset
    
    validation_split = 0.30
    batch_size = 1000
    parquet_file = f"{input_file}.parquet"
    
    # hf_dataset = load_dataset("parquet", data_files=parquet_file, split="train")
    # ds_splits = hf_dataset.train_test_split(test_size=validation_split, seed=42, shuffle=True)

    # train_ds = ds_splits["train"].with_format(type='torch', columns=['features', 'Label'])
    # val_ds = ds_splits["test"].with_format(type='torch', columns=['features', 'Label'])    

    # gen_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
    # gen_validation = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)

    hf_dataset = load_dataset("parquet", data_files=parquet_file, split="train", streaming=True)
    hf_dataset = hf_dataset.shuffle(seed=42, buffer_size=dataset_size/3)

    val_size = int(dataset_size * 0.3)
    train_size = dataset_size - val_size

    val_ds = list(hf_dataset.take(val_size))
    train_ds = list(hf_dataset.skip(val_size))

    from datasets import Dataset
    train_ds = Dataset.from_list(train_ds).with_format("torch")
    val_ds = Dataset.from_list(val_ds).with_format("torch")

    gen_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    gen_validation = DataLoader(val_ds, batch_size=batch_size, shuffle=True)


elif Type == "root":    
    # generate batches for the training 
    print("####### ROOT DataLoader ###############")
    # gen_train, gen_validation =  ROOT.TMVA.Experimental.CreatePyTorchGenerators(df,
    #                                                                             Batch_size,
    #                                                                             chunk_size,
    #                                                                             block_size,
    #                                                                             columns = training_features,
    #                                                                             target = training_label,
    #                                                                             validation_split = 0.30,
    #                                                                             shuffle = True,
    #                                                                             drop_remainder = True,
    #                                                                             set_seed = 0)
    gen_train, gen_validation = ROOT.Experimental.ML.RDataLoader(dsid_dfs,
                                                                 Batch_size,
                                                                 batches_in_memory = batches_in_memory,
                                                                 columns = training_features,
                                                                 target = training_label,
                                                                 validation_split = 0.30,
                                                                 shuffle = True,
                                                                 drop_remainder = False,
                                                                 set_seed = 0)
                                                                 

# PNN model configureation
neuron = 60

model = PNN(num_features, neuron)
num_neg = SM_size
num_pos = SUSY_size
w_0 = (SM_size + SUSY_size)/(2*SM_size)
w_1 = (SM_size + SUSY_size)/(2*SUSY_size)
weight = torch.tensor([0.1, 0.9])
pos_weight = torch.tensor([(num_neg / num_pos)/imb])
Weights = torch.tensor([w_0, w_1])
print(w_0, w_1)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

# calculate imbalance in dataset
# SUSY_events = 0
# training_events = 0
# for i, batch in enumerate(gen_train.AsTorch() if Type == "root" else gen_train):
#     if Type == "parquet":
#         x_train = batch["features"]
#         y_train = batch["Label"].unsqueeze(1)
#     else:
#         x_train, y_train = batch

#     SUSY_events += y_train.sum().item()
#     training_events += len(y_train)
# print("Training events ", training_events)
# print(f"SUSY fraction in training dataset: {SUSY_events / training_events}")
# 

Loss_training = []
Loss_validation = []

Accuracy_training = []
Accuracy_validation = []

AUC_training = []
AUC_validation = []

PR_AUC_training = []
PR_AUC_validation = []

Precision_training = []
Precision_validation = []

Recall_training = []
Recall_validation = []

epoch_times = []

early_stopping = EarlyStopping(patience=100, verbose=True)
for epoch in range(epochs):
    print("############################################")
    print(f"################ Epcoch {epoch+1} ################")
    start = time.time()

    ##############################################
    # Training
    ##############################################
    
    # Loop through the training set and train model
    Accuracy = 0
    Loss = 0
    batches = 0
    model.train()

    Outputs = []
    Labels = []

    for i, batch in enumerate(gen_train.AsTorch() if Type == "root" else gen_train):
        if Type == "parquet":
            inputs = batch["features"]
            labels = batch["Label"].unsqueeze(1)
        else:
            inputs, labels = batch

        optimizer.zero_grad()        
        logits = model(inputs)
        loss = loss_fn(logits, labels)
            
        # improve model

        loss.backward()
        optimizer.step()
 
        # Calculate accuracy
        accuracy = calc_accuracy(labels, torch.sigmoid(logits))
        f1_metric = BinaryF1Score()
        f1 = f1_metric(torch.sigmoid(logits), labels)

        batch_outputs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        batch_labels = labels.detach().cpu().numpy().flatten()
    
        Outputs.append(batch_outputs)
        Labels.append(batch_labels)        
        
        Loss += loss.item()
        Accuracy += accuracy.item()        
        batches += 1
        
    # calculate average 
    Loss_avg = Loss / batches
    Accuracy_avg = Accuracy / batches
    Outputs = np.concatenate(Outputs)
    Labels = np.concatenate(Labels)
    AUC = roc_auc_score(Labels, Outputs)
    precision, recall, _ = precision_recall_curve(Labels, Outputs)
    PR_AUC = auc(recall, precision)

    preds = (Outputs > 0.5).astype(int)
    precision = precision_score(Labels, preds)
    recall = recall_score(Labels, preds)        

    print(f"Training => AUC: {AUC:.3f} PR AUC: {PR_AUC:.3f} loss: {Loss_avg:.3f}")
    print(f"Epoch Precision: {precision:.4f}")
    print(f"Epoch Recall: {recall:.4f}")
    
    Loss_training.append(Loss_avg)
    Accuracy_training.append(Accuracy_avg)
    AUC_training.append(AUC)
    PR_AUC_training.append(PR_AUC)
    Precision_training.append(precision)
    Recall_training.append(recall)

    ##############################################
    # Validation
    ##############################################

    # Evaluate the model on the validation set
    with torch.no_grad():
        Accuracy = 0
        Loss = 0
        batches = 0
        
        Outputs = []
        Labels = []

        model.eval()
        for i, batch in enumerate(gen_validation.AsTorch() if Type == "root" else gen_validation):
            if Type == "parquet":
                inputs = batch["features"]
                labels = batch["Label"].unsqueeze(1)
            else:
                inputs, labels = batch

            logits = model(inputs)
            # Calculate accuracy
            accuracy = calc_accuracy(labels, torch.sigmoid(logits))
            loss = loss_fn(logits, labels)
            f1_metric = BinaryF1Score()
            f1 = f1_metric(torch.sigmoid(logits), labels)
            
            batch_outputs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            batch_labels = labels.detach().cpu().numpy().flatten()
    
            Outputs.append(batch_outputs)
            Labels.append(batch_labels)        
            
            Loss += loss.item()
            Accuracy += accuracy.item()        
            batches += 1
            
        # calculate average 
        Loss_avg = Loss / batches
        Accuracy_avg = Accuracy / batches
        Outputs = np.concatenate(Outputs)
        Labels = np.concatenate(Labels)
        AUC = roc_auc_score(Labels, Outputs)

        precision, recall, _ = precision_recall_curve(Labels, Outputs)
        PR_AUC = auc(recall, precision)

        preds = (Outputs > 0.5).astype(int)
        precision = precision_score(Labels, preds)
        recall = recall_score(Labels, preds)        

        print(f"Validation => AUC: {AUC:.3f} PR AUC: {PR_AUC:.3f} loss: {Loss_avg:.3f}")
        print(f"Epoch Precision: {precision:.4f}")
        print(f"Epoch Recall: {recall:.4f}")
        Loss_validation.append(Loss_avg)
        Accuracy_validation.append(Accuracy_avg)
        AUC_validation.append(AUC)
        PR_AUC_validation.append(PR_AUC)
        Precision_validation.append(precision)
        Recall_validation.append(recall)
        
    early_stopping(Loss_avg, model)

    epoch_times.append(time.time() - start)
    print(f"Epoch {epoch+1} completed in {epoch_times[-1]:.2f} seconds")

    if early_stopping.early_stop:
        print("Early stopping triggered")
        break


times_file = f"../results/epoch_times_{Type}.csv"

with open(times_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "time_seconds"])
    for i, t in enumerate(epoch_times):
        writer.writerow([i + 1, t])



model = PNN(num_features, neuron)
model.load_state_dict(torch.load(f"../PNN_models/checkpoint_m:{SUSY_model}_lep:{lepton}_lr:{lr}_wd:{wd}_.pt", weights_only=True))
model.eval()

Outputs_training = []
Labels_training = []

for i, batch in enumerate(gen_train.AsTorch() if Type == "root" else gen_train):
    if Type == "parquet":
        inputs = batch["features"]
        labels = batch["Label"].unsqueeze(1)
    else:
        inputs, labels = batch

    model.eval()
    logits = model(inputs)
    outputs = torch.sigmoid(logits)

    Outputs = torch.flatten(outputs).detach().cpu().numpy()
    Labels = torch.flatten(labels).detach().cpu().numpy()
    
    # Concatenate numpy arrays
    Outputs_training = np.concatenate((Outputs_training, Outputs))
    Labels_training = np.concatenate((Labels_training, Labels))            

Outputs_validation = []
Labels_validation = []
for i, batch in enumerate(gen_validation.AsTorch() if Type == "root" else gen_validation):
    if Type == "parquet":
        inputs = batch["features"]
        labels = batch["Label"].unsqueeze(1)
    else:
        inputs, labels = batch

    model.eval()
    logits = model(inputs)
    outputs = torch.sigmoid(logits)

    Outputs = torch.flatten(outputs).detach().cpu().numpy()
    Labels = torch.flatten(labels).detach().cpu().numpy()
    
    # Concatenate numpy arrays
    Outputs_validation = np.concatenate((Outputs_validation, Outputs))
    Labels_validation = np.concatenate((Labels_validation, Labels))            

configuration = f"{imb}_{lr}_{wd}_{Type}"    

df = ROOT.RDF.FromNumpy({"outputs": Outputs_training, "labels": Labels_training})
df.Define("ratio", "outputs / labels").Snapshot("tree", f"../results/outputs_training_{configuration}.root")

df = ROOT.RDF.FromNumpy({"outputs": Outputs_validation, "labels": Labels_validation})
df.Define("ratio", "outputs / labels").Snapshot("tree", f"../results/outputs_validation_{configuration}.root")


x, y = np.array(Loss_training, dtype=np.float64), np.array(Loss_validation, dtype=np.float64)
df = ROOT.RDF.FromNumpy({"loss_training": x, "loss_validation": y})
df.Define("ratio", "loss_training / loss_validation").Snapshot("tree", "../results/loss.root")

Accuracy_training_np = np.array(Accuracy_training, dtype=np.float64)
Loss_training_np = np.array(Loss_training, dtype=np.float64)
AUC_training_np = np.array(AUC_training, dtype=np.float64)
PR_AUC_training_np = np.array(PR_AUC_training, dtype=np.float64)
Precision_training_np = np.array(Precision_training, dtype=np.float64)
Recall_training_np = np.array(Recall_training, dtype=np.float64)

Accuracy_validation_np = np.array(Accuracy_validation, dtype=np.float64)
Loss_validation_np = np.array(Loss_validation, dtype=np.float64)
AUC_validation_np = np.array(AUC_validation, dtype=np.float64)
PR_AUC_validation_np = np.array(PR_AUC_validation, dtype=np.float64)
Precision_validation_np = np.array(Precision_validation, dtype=np.float64)
Recall_validation_np = np.array(Recall_validation, dtype=np.float64)

print(Accuracy_training_np)
print(Loss_training_np)
print(AUC_training_np)
print(Loss_training_np)
print(Precision_training_np)
print(Recall_training_np)
print(Accuracy_validation_np)
print(Loss_validation_np)
print(AUC_validation_np)
print(PR_AUC_validation_np)
print(Loss_validation_np)
print(Precision_validation_np)
print(Recall_validation_np)

columns = ["Accuracy_training",
           "Loss_training",
           "AUC_training",
           "PR_AUC_training",
           "Precision_training",
           "Recall_training",
           "Accuracy_validation",
           "Loss_validation",
           "AUC_validation",
           "PR_AUC_validation",
           "Precision_validation",
           "Recall_validation"]

df = ROOT.RDF.FromNumpy({"Accuracy_training": Accuracy_training_np,
                         "Loss_training": Loss_training_np,
                         "AUC_training": AUC_training_np,
                         "PR_AUC_training": PR_AUC_training_np,
                         "Precision_training": Precision_training_np,
                         "Recall_training": Recall_training_np,                         
                         "Accuracy_validation": Accuracy_validation_np,                         
                         "Loss_validation": Loss_validation_np,
                         "AUC_validation": AUC_validation_np,
                         "PR_AUC_validation": PR_AUC_validation_np,
                         "Precision_validation": Precision_validation_np,
                         "Recall_validation": Recall_validation_np})                         

for i in columns:
    df = df.Define(f"{i}_2", f"2 * {i}")
    
df.Snapshot("tree", f"../results/training_{configuration}.root", columns)
