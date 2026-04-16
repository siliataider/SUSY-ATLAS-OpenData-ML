import ROOT
import numpy as np

df_sm = ROOT.RDataFrame("CollectionTree", "/home/siliataider/Documents/root/data_loader_bench/SUSY-ATLAS-OpenData-ML/data/SM_SUSY_C1C1_mu_mll-15_MET_both_jets_sm.root")

dsids = df_sm.AsNumpy(["dsid"])["dsid"]
unique_dsids, counts = np.unique(dsids, return_counts=True)

# print("\nDSID SM\tCount")
# for dsid, count in zip(unique_dsids, counts):
#     print(f"{dsid}\t{count}")

# print(len(unique_dsids))

for dsid in unique_dsids:
    df_sm.Filter(f"dsid == {dsid}").Snapshot("CollectionTree", f"/home/siliataider/Documents/root/data_loader_bench/SUSY-ATLAS-OpenData-ML/data/dsid_files/dsid_{dsid}_sm.root")


df_susy = ROOT.RDataFrame("CollectionTree", "/home/siliataider/Documents/root/data_loader_bench/SUSY-ATLAS-OpenData-ML/data/SM_SUSY_C1C1_mu_mll-15_MET_both_jets_susy.root")

dsids = df_susy.AsNumpy(["dsid"])["dsid"]
unique_dsids, counts = np.unique(dsids, return_counts=True)

for dsid in unique_dsids:
    df_susy.Filter(f"dsid == {dsid}").Snapshot("CollectionTree", f"/home/siliataider/Documents/root/data_loader_bench/SUSY-ATLAS-OpenData-ML/data/dsid_files/dsid_{dsid}_susy.root")

# print("\nDSID SUSY\tCount")
# for dsid, count in zip(unique_dsids, counts):
#     print(f"{dsid}\t{count}")

# print(len(unique_dsids))

"""
59 unique dsids for SM, 134 for SUSY

"""