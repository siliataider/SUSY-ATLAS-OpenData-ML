import ROOT
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc

ROOT.gStyle.SetOptStat(0)

parser = argparse.ArgumentParser()
parser.add_argument('--imb', type=float, default=50, help='Imbalance factor to scale down class weight')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='Weight decay')
parser.add_argument('--Type', type=str, choices=['training', 'validation'], help='Type of dataset')
args = parser.parse_args()

imb = args.imb
lr = args.lr
wd = args.wd
Type = args.Type
configuration = f"{imb}_{lr}_{wd}"

# Load ROOT data
df_root = ROOT.RDataFrame("tree", f"../results/outputs_{Type}_{configuration}_root.root")
print("Total events (ROOT):", df_root.Count().GetValue())
output_root = df_root.AsNumpy(["outputs", "labels"])
scores_root = output_root["outputs"]
labels_root = output_root["labels"]
fpr_root, tpr_root, _ = roc_curve(labels_root, scores_root)
roc_auc_root = auc(fpr_root, tpr_root)

# Load CSV data
df_csv = ROOT.RDataFrame("tree", f"../results/outputs_{Type}_{configuration}_csv.root")
print("Total events (CSV):", df_csv.Count().GetValue())
output_csv = df_csv.AsNumpy(["outputs", "labels"])
scores_csv = output_csv["outputs"]
labels_csv = output_csv["labels"]
fpr_csv, tpr_csv, _ = roc_curve(labels_csv, scores_csv)
roc_auc_csv = auc(fpr_csv, tpr_csv)

# Load Parquet data
df_parquet = ROOT.RDataFrame("tree", f"../results/outputs_{Type}_{configuration}_parquet.root")
print("Total events (Parquet):", df_parquet.Count().GetValue())
output_parquet = df_parquet.AsNumpy(["outputs", "labels"])
scores_parquet = output_parquet["outputs"]
labels_parquet = output_parquet["labels"]
fpr_parquet, tpr_parquet, _ = roc_curve(labels_parquet, scores_parquet)
roc_auc_parquet = auc(fpr_parquet, tpr_parquet)

# Create TGraphs
graph_root = ROOT.TGraph(len(fpr_root), np.array(fpr_root, dtype='float64'), np.array(tpr_root, dtype='float64'))
graph_csv = ROOT.TGraph(len(fpr_csv), np.array(fpr_csv, dtype='float64'), np.array(tpr_csv, dtype='float64'))
graph_parquet = ROOT.TGraph(len(fpr_parquet), np.array(fpr_parquet, dtype='float64'), np.array(tpr_parquet, dtype='float64'))

# Smooth the ROC graphs
smoother_root = ROOT.TGraphSmooth()
smoother_csv = ROOT.TGraphSmooth()
smoother_parquet = ROOT.TGraphSmooth()

smooth_graph_root = smoother_root.SmoothSuper(graph_root, "linear")
smooth_graph_csv = smoother_csv.SmoothSuper(graph_csv, "linear")
smooth_graph_parquet = smoother_parquet.SmoothSuper(graph_parquet, "linear")

# Modern color palette (consistent with other plots)
color_root    = ROOT.TColor.GetColor("#e63946")  # Red
color_csv     = ROOT.TColor.GetColor("#2a9d8f")  # Teal
color_parquet = ROOT.TColor.GetColor("#7b2cbf")  # Purple

lineWidth = 3

# Style ROOT graph
smooth_graph_root.SetLineWidth(lineWidth)
smooth_graph_root.SetLineColor(color_root)
smooth_graph_root.SetLineStyle(1)

# Style CSV graph
smooth_graph_csv.SetLineWidth(lineWidth)
smooth_graph_csv.SetLineColor(color_csv)
smooth_graph_csv.SetLineStyle(1)

# Style Parquet graph
smooth_graph_parquet.SetLineWidth(lineWidth)
smooth_graph_parquet.SetLineColor(color_parquet)
smooth_graph_parquet.SetLineStyle(1)

# Global style settings
labelSize = 30
ROOT.gStyle.SetLabelFont(43, "XYZ")
ROOT.gStyle.SetLabelSize(labelSize, "xyz")
ROOT.gStyle.SetTitleFont(43, "XYZ")
ROOT.gStyle.SetTitleSize(labelSize, "xyz")
ROOT.gStyle.SetTextFont(43)
ROOT.gStyle.SetTextSize(labelSize)

ROOT.gROOT.SetBatch(True)
c = ROOT.TCanvas("c", "Smoothed ROC Curve", 800, 800)
c.SetFillColor(ROOT.kWhite)
c.SetGrid()
c.SetTopMargin(0.05)
c.SetLeftMargin(0.15)
c.SetRightMargin(0.05)
c.SetBottomMargin(0.15)

# Draw graphs
smooth_graph_root.SetTitle(";False Positive Rate;True Positive Rate")
smooth_graph_root.Draw("AL")
smooth_graph_root.GetXaxis().SetLimits(-0.01, 1.01)
smooth_graph_root.GetYaxis().SetRangeUser(-0.01, 1.01)
smooth_graph_root.GetXaxis().SetTitleOffset(1.2)
smooth_graph_root.GetYaxis().SetTitleOffset(1.4)

smooth_graph_csv.Draw("L SAME")
smooth_graph_parquet.Draw("L SAME")

# Diagonal reference line (random classifier)
line = ROOT.TLine(0, 0, 1, 1)
line.SetLineStyle(2)
line.SetLineColor(ROOT.kGray+2)
line.SetLineWidth(2)
line.Draw("same")

# Legend
legend = ROOT.TLegend(0.2, 0.25, 0.9, 0.45)
legend.SetBorderSize(0)
legend.SetFillStyle(1001)
legend.SetFillColor(ROOT.kWhite)
legend.SetTextFont(43)
legend.SetTextSize(22)
legend.AddEntry(smooth_graph_root, f"ROOT (AUC = {roc_auc_root:.4f})", "l")
legend.AddEntry(smooth_graph_csv, f"Pandas-PyTorch/CSV (AUC = {roc_auc_csv:.4f})", "l")
legend.AddEntry(smooth_graph_parquet, f"HuggingFace-PyTorch/Parquet (AUC = {roc_auc_parquet:.4f})", "l")
legend.AddEntry(line, "Random classifier", "l")
legend.Draw()

c.Update()
c.SaveAs(f"../plots/roc_{Type}_{configuration}.pdf")