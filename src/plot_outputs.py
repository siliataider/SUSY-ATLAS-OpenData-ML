import ROOT
import argparse
ROOT.gStyle.SetOptStat(0)

parser = argparse.ArgumentParser(description="Process some training parameters.")
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
df = ROOT.RDataFrame("tree", f"../results/outputs_{Type}_{configuration}_root.root")    
print(df.Count().GetValue())
output = df.AsNumpy(["outputs", "labels"])
outputs = output["outputs"]
labels = output["labels"]
outputs_labels_sig = [p for p, t in zip(outputs, labels) if t == 1]
outputs_labels_bkg = [p for p, t in zip(outputs, labels) if t == 0]
events = len(outputs)

sig_events = len(outputs_labels_sig)
sig_ratio = sig_events / events

bkg_events = len(outputs_labels_bkg)
bkg_ratio = bkg_events / events

n_bins = 50
alpha = 0.4

labelSize = 37
ROOT.gStyle.SetLabelFont(43,"XYZ")
ROOT.gStyle.SetLabelSize(labelSize,"xyz")
ROOT.gStyle.SetTitleFont(43,"XYZ")
ROOT.gStyle.SetTitleSize(labelSize,"xyz")
ROOT.gStyle.SetTextFont(43)
ROOT.gStyle.SetTextSize(labelSize)

hist_sig = ROOT.TH1F(f"hist_sig", ";Model output;Events", n_bins, 0, 1)
hist_sig.SetFillColorAlpha(ROOT.kRed+1, alpha)
hist_sig.SetFillStyle(1001)
hist_sig.SetLineColor(ROOT.kRed+2)
hist_sig.SetLineWidth(2)
hist_sig.SetMinimum(100)
hist_sig.SetMaximum(0.2*1000000)

hist_bkg = ROOT.TH1F(f"hist_bkg", ";Model output;Events", n_bins, 0, 1)
hist_bkg.SetFillColorAlpha(ROOT.kBlue, alpha)
hist_bkg.SetFillStyle(1001)
hist_bkg.SetLineColor(ROOT.kBlue+1)
hist_bkg.SetLineWidth(2)

for i in outputs_labels_sig:
    hist_sig.Fill(i)

for i in outputs_labels_bkg:
    hist_bkg.Fill(i)
    
ROOT.gROOT.SetBatch(True)
c = ROOT.TCanvas("c", "Histogram Canvas", 800, 800)
c.SetGrid()
c.SetLeftMargin(0.15)
c.SetRightMargin(0.01)
c.SetBottomMargin(0.15)
c.SetTopMargin(0.02)

ROOT.gStyle.SetLegendFont(42)
ROOT.gPad.SetLogy()

hist_sig.Draw("HIST")
hist_sig.GetXaxis().SetTitleOffset(1.3)
hist_bkg.Draw("HIST SAME")

legend = ROOT.TLegend(0.55, 0.80, 0.95, 0.93)
legend.AddEntry(hist_bkg, f"SM ({round(bkg_ratio * 100, 1)}%)", "F")
legend.AddEntry(hist_sig, f"SUSY ({round(sig_ratio * 100, 1)}%)", "F")
legend.SetTextFont(43)
legend.SetTextSize(labelSize)
legend.SetBorderSize(1)
legend.Draw()

c.Update()
c.SaveAs(f"../plots/outputs_{configuration}_{Type}_root.pdf")
