import sys
import ROOT
import numpy as np
import argparse

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

parser = argparse.ArgumentParser(description="Process some training parameters.")
parser.add_argument('--imb', type=float, default=2.0, help='Imbalance factor to scale down class weight')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
parser.add_argument('--var', type=str, choices=['Loss', 'AUC', 'Time'], help='Number of epochs')
args = parser.parse_args()

imb = args.imb
lr = args.lr
wd = args.wd
var = args.var

configuration = f"{imb}_{lr}_{wd}"    
df_root = ROOT.RDataFrame("tree", f"../results/training_{configuration}_root.root")
df_csv = ROOT.RDataFrame("tree", f"../results/training_{configuration}_csv.root")
df_parquet = ROOT.RDataFrame("tree", f"../results/training_{configuration}_parquet.root")

df_columns_root = df_root.AsNumpy(columns)
df_columns_csv = df_csv.AsNumpy(columns)
df_columns_parquet = df_parquet.AsNumpy(columns)

Accuracy_training_root = df_columns_root["Accuracy_training"]
Loss_training_root = df_columns_root["Loss_training"]
AUC_training_root = df_columns_root["AUC_training"]
PR_AUC_training_root = df_columns_root["PR_AUC_training"]
Loss_training_root = df_columns_root["Loss_training"]
Precision_training_root = df_columns_root["Precision_training"]
Recall_training_root = df_columns_root["Recall_training"]

Accuracy_validation_root = df_columns_root["Accuracy_validation"]
Loss_validation_root = df_columns_root["Loss_validation"]
AUC_validation_root = df_columns_root["AUC_validation"]
PR_AUC_validation_root = df_columns_root["PR_AUC_validation"]
Loss_validation_root = df_columns_root["Loss_validation"]
Precision_validation_root = df_columns_root["Precision_validation"]
Recall_validation_root = df_columns_root["Recall_validation"]

Accuracy_training_csv = df_columns_csv["Accuracy_training"]
Loss_training_csv = df_columns_csv["Loss_training"]
AUC_training_csv = df_columns_csv["AUC_training"]
PR_AUC_training_csv = df_columns_csv["PR_AUC_training"]
Loss_training_csv = df_columns_csv["Loss_training"]
Precision_training_csv = df_columns_csv["Precision_training"]
Recall_training_csv = df_columns_csv["Recall_training"]

Accuracy_validation_csv = df_columns_csv["Accuracy_validation"]
Loss_validation_csv = df_columns_csv["Loss_validation"]
AUC_validation_csv = df_columns_csv["AUC_validation"]
PR_AUC_validation_csv = df_columns_csv["PR_AUC_validation"]
Loss_validation_csv = df_columns_csv["Loss_validation"]
Precision_validation_csv = df_columns_csv["Precision_validation"]
Recall_validation_csv = df_columns_csv["Recall_validation"]

Accuracy_training_parquet = df_columns_parquet["Accuracy_training"]
Loss_training_parquet = df_columns_parquet["Loss_training"]
AUC_training_parquet = df_columns_parquet["AUC_training"]
PR_AUC_training_parquet = df_columns_parquet["PR_AUC_training"]
Loss_training_parquet = df_columns_parquet["Loss_training"]
Precision_training_parquet = df_columns_parquet["Precision_training"]
Recall_training_parquet = df_columns_parquet["Recall_training"]

Accuracy_validation_parquet = df_columns_parquet["Accuracy_validation"]
Loss_validation_parquet = df_columns_parquet["Loss_validation"]
AUC_validation_parquet = df_columns_parquet["AUC_validation"]
PR_AUC_validation_parquet = df_columns_parquet["PR_AUC_validation"]
Loss_validation_parquet = df_columns_parquet["Loss_validation"]
Precision_validation_parquet = df_columns_parquet["Precision_validation"]
Recall_validation_parquet = df_columns_parquet["Recall_validation"]

N = len(Accuracy_training_root)
X = np.linspace(1, N, N)

graph_Accuracy_training_root = ROOT.TGraph(N, X, np.array(Accuracy_training_root))
graph_Loss_training_root = ROOT.TGraph(N, X, np.array(Loss_training_root))
graph_AUC_training_root = ROOT.TGraph(N, X, np.array(AUC_training_root))
graph_PR_AUC_training_root = ROOT.TGraph(N, X, np.array(PR_AUC_training_root))
graph_Precision_training_root = ROOT.TGraph(N, X, np.array(Precision_training_root))
graph_Recall_training_root = ROOT.TGraph(N, X, np.array(Recall_training_root))

graph_Accuracy_validation_root = ROOT.TGraph(N, X, np.array(Accuracy_validation_root))
graph_Loss_validation_root = ROOT.TGraph(N, X, np.array(Loss_validation_root))
graph_AUC_validation_root = ROOT.TGraph(N, X, np.array(AUC_validation_root))
graph_PR_AUC_validation_root = ROOT.TGraph(N, X, np.array(PR_AUC_validation_root))
graph_Precision_validation_root = ROOT.TGraph(N, X, np.array(Precision_validation_root))
graph_Recall_validation_root = ROOT.TGraph(N, X, np.array(Recall_validation_root))

graph_Accuracy_training_csv = ROOT.TGraph(N, X, np.array(Accuracy_training_csv))
graph_Loss_training_csv = ROOT.TGraph(N, X, np.array(Loss_training_csv))
graph_AUC_training_csv = ROOT.TGraph(N, X, np.array(AUC_training_csv))
graph_PR_AUC_training_csv = ROOT.TGraph(N, X, np.array(PR_AUC_training_csv))
graph_Precision_training_csv = ROOT.TGraph(N, X, np.array(Precision_training_csv))
graph_Recall_training_csv = ROOT.TGraph(N, X, np.array(Recall_training_csv))

graph_Accuracy_validation_csv = ROOT.TGraph(N, X, np.array(Accuracy_validation_csv))
graph_Loss_validation_csv = ROOT.TGraph(N, X, np.array(Loss_validation_csv))
graph_AUC_validation_csv = ROOT.TGraph(N, X, np.array(AUC_validation_csv))
graph_PR_AUC_validation_csv = ROOT.TGraph(N, X, np.array(PR_AUC_validation_csv))
graph_Precision_validation_csv = ROOT.TGraph(N, X, np.array(Precision_validation_csv))
graph_Recall_validation_csv = ROOT.TGraph(N, X, np.array(Recall_validation_csv))

graph_Accuracy_training_parquet = ROOT.TGraph(N, X, np.array(Accuracy_training_parquet))
graph_Loss_training_parquet = ROOT.TGraph(N, X, np.array(Loss_training_parquet))
graph_AUC_training_parquet = ROOT.TGraph(N, X, np.array(AUC_training_parquet))
graph_PR_AUC_training_parquet = ROOT.TGraph(N, X, np.array(PR_AUC_training_parquet))
graph_Precision_training_parquet = ROOT.TGraph(N, X, np.array(Precision_training_parquet))
graph_Recall_training_parquet = ROOT.TGraph(N, X, np.array(Recall_training_parquet))

graph_Accuracy_validation_parquet = ROOT.TGraph(N, X, np.array(Accuracy_validation_parquet))
graph_Loss_validation_parquet = ROOT.TGraph(N, X, np.array(Loss_validation_parquet))
graph_AUC_validation_parquet = ROOT.TGraph(N, X, np.array(AUC_validation_parquet))
graph_PR_AUC_validation_parquet = ROOT.TGraph(N, X, np.array(PR_AUC_validation_parquet))
graph_Precision_validation_parquet = ROOT.TGraph(N, X, np.array(Precision_validation_parquet))
graph_Recall_validation_parquet = ROOT.TGraph(N, X, np.array(Recall_validation_parquet))

labelSize = 30
ROOT.gStyle.SetLabelFont(43,"XYZ")
ROOT.gStyle.SetLabelSize(labelSize,"xyz")
ROOT.gStyle.SetTitleFont(43,"XYZ")
ROOT.gStyle.SetTitleSize(labelSize,"xyz")
ROOT.gStyle.SetTextFont(43)
ROOT.gStyle.SetTextSize(labelSize)

ROOT.gROOT.SetBatch(True)
c = ROOT.TCanvas("c", "Histogram Canvas", 800, 800)
c.SetFillColor(ROOT.kWhite)
c.SetGrid()
c.SetLeftMargin(0.15)
c.SetRightMargin(0.05)
c.SetBottomMargin(0.15)
c.SetTopMargin(0.05)

if var == "Loss":
    graph1 = graph_Loss_training_root
    graph2 = graph_Loss_validation_root
    graph3 = graph_Loss_training_csv
    graph4 = graph_Loss_validation_csv
    graph5 = graph_Loss_training_parquet
    graph6 = graph_Loss_validation_parquet

elif var == "AUC":
    graph1 = graph_AUC_training_root
    graph2 = graph_AUC_validation_root
    graph3 = graph_AUC_training_csv
    graph4 = graph_AUC_validation_csv
    graph5 = graph_AUC_training_parquet
    graph6 = graph_AUC_validation_parquet

elif var == 'Time':
    import csv
    
    def read_epoch_times(filepath):
        epochs, times = [], []
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(float(row["epoch"]))
                times.append(float(row["time_seconds"]))
        return np.array(epochs), np.array(times)
    
    epochs_root, times_root = read_epoch_times("../results/epoch_times_root.csv")
    epochs_csv,  times_csv  = read_epoch_times("../results/epoch_times_csv.csv")
    epochs_parquet, times_parquet = read_epoch_times("../results/epoch_times_parquet.csv")
    
    N_root = len(epochs_root)
    N_csv  = len(epochs_csv)
    N_parquet = len(epochs_parquet)
    
    graph1 = ROOT.TGraph(N_csv,  epochs_csv,  times_csv)
    graph2 = ROOT.TGraph(N_parquet, epochs_parquet, times_parquet)
    graph3 = ROOT.TGraph(N_root, epochs_root, times_root)
    
    color1 = ROOT.TColor.GetColor("#e63946")
    color2 = ROOT.TColor.GetColor("#2a9d8f")
    color3 = ROOT.TColor.GetColor("#7b2cbf")
    
    graph1.SetMarkerStyle(20)
    graph1.SetMarkerSize(1.2)
    graph1.SetMarkerColor(color1)
    graph1.SetLineColor(color1)
    graph1.SetLineWidth(2)
    
    graph2.SetMarkerStyle(21)
    graph2.SetMarkerSize(1.2)
    graph2.SetMarkerColor(color2)
    graph2.SetLineColor(color2)
    graph2.SetLineWidth(2)
    
    graph3.SetMarkerStyle(22)
    graph3.SetMarkerSize(1.4)
    graph3.SetMarkerColor(color3)
    graph3.SetLineColor(color3)
    graph3.SetLineWidth(2)
    
    y_max = max(times_root.max(), times_csv.max(), times_parquet.max()) * 1.15
    graph1.SetMinimum(0)
    graph1.SetMaximum(y_max)
    
    graph1.Draw("APL")
    graph2.Draw("PL same")
    graph3.Draw("PL same")
    
    legend = ROOT.TLegend(0.35, 0.65, 0.88, 0.8)
    legend.SetBorderSize(0)
    legend.SetFillStyle(1001)
    legend.SetFillColor(ROOT.kWhite)
    legend.SetTextFont(43)
    legend.SetTextSize(25)
    legend.AddEntry(graph1, "Pandas-PyTorch / CSV", "lp")
    legend.AddEntry(graph2, "HuggingFace-PyTorch / Parquet", "lp")
    legend.AddEntry(graph3, "ROOT", "lp")
    legend.Draw()
    
    graph1.SetTitle(";Epoch;Time (s)")
    graph1.GetXaxis().SetTitleOffset(1.2)
    graph1.GetYaxis().SetTitleOffset(1.2)
    
    c.Update()
    c.SaveAs(f"../plots/Time_{configuration}_both.pdf")
    sys.exit(0)

# Modern color palette: 3 distinct colors for 3 methods
color_root    = ROOT.TColor.GetColor("#e63946")  # Red
color_csv     = ROOT.TColor.GetColor("#2a9d8f")  # Teal
color_parquet = ROOT.TColor.GetColor("#7b2cbf")  # Purple

lineWidth = 3

# ROOT: Training (solid) & Validation (dashed)
graph1.SetLineWidth(lineWidth)
graph1.SetLineColor(color_root)
graph1.SetLineStyle(1)  # Solid
graph1.SetMarkerStyle(20)
graph1.SetMarkerSize(0.9)
graph1.SetMarkerColor(color_root)

graph2.SetLineWidth(lineWidth)
graph2.SetLineColor(color_root)
graph2.SetLineStyle(2)  # Dashed
graph2.SetMarkerStyle(24)  # Open circle
graph2.SetMarkerSize(0.9)
graph2.SetMarkerColor(color_root)

# CSV/PyTorch: Training (solid) & Validation (dashed)
graph3.SetLineWidth(lineWidth)
graph3.SetLineColor(color_csv)
graph3.SetLineStyle(1)
graph3.SetMarkerStyle(21)
graph3.SetMarkerSize(0.9)
graph3.SetMarkerColor(color_csv)

graph4.SetLineWidth(lineWidth)
graph4.SetLineColor(color_csv)
graph4.SetLineStyle(2)
graph4.SetMarkerStyle(25)  # Open square
graph4.SetMarkerSize(0.9)
graph4.SetMarkerColor(color_csv)

# Parquet: Training (solid) & Validation (dashed)
graph5.SetLineWidth(lineWidth)
graph5.SetLineColor(color_parquet)
graph5.SetLineStyle(1)
graph5.SetMarkerStyle(22)
graph5.SetMarkerSize(1.0)
graph5.SetMarkerColor(color_parquet)

graph6.SetLineWidth(lineWidth)
graph6.SetLineColor(color_parquet)
graph6.SetLineStyle(2)
graph6.SetMarkerStyle(26)  # Open triangle
graph6.SetMarkerSize(1.0)
graph6.SetMarkerColor(color_parquet)

# Draw with points and lines
graph1.Draw("APL")
graph2.Draw("PL same")
graph3.Draw("PL same")
graph4.Draw("PL same")
graph5.Draw("PL same")
graph6.Draw("PL same")

# Clean legend with grouping
legend = ROOT.TLegend(0.25, 0.35, 0.93, 0.7)
if var == "Loss":
    legend = ROOT.TLegend(0.25, 0.55, 0.93, 0.93)

legend.SetBorderSize(0)
legend.SetFillStyle(1001)
legend.SetFillColor(ROOT.kWhite)
legend.SetTextFont(43)
legend.SetTextSize(22)
legend.SetNColumns(1)

legend.AddEntry(graph1, "ROOT - Training", "lp")
legend.AddEntry(graph2, "ROOT - Validation", "lp")
legend.AddEntry(graph3, "Pandas-PyTorch/CSV - Training", "lp")
legend.AddEntry(graph4, "Pandas-PyTorch/CSV - Validation", "lp")
legend.AddEntry(graph5, "HuggingFace-PyTorch/Parquet - Training", "lp")
legend.AddEntry(graph6, "HuggingFace-PyTorch/Parquet - Validation", "lp")
legend.Draw()

graph1.SetTitle(f";Epoch;{var}")
graph1.GetXaxis().SetTitleOffset(1.2)
graph1.GetYaxis().SetTitleOffset(1.4)

c.Update()
c.SaveAs(f"../plots/{var}_{configuration}_both.pdf")