import ROOT
import pathlib

file = ROOT.TFile("/home/siliataider/Documents/root/data_loader_bench/SUSY-ATLAS-OpenData-ML/data/SM_SUSY_C1C1_mu_mll-15_MET_both_jets.root")

# print number of clusters
tree = file.Get("CollectionTree")
n_entries = tree.GetEntries()
print("Number of events: ", n_entries)

it = tree.GetClusterIterator(0)

cluster_sizes = []
n_clusters = 0

entry = it.Next()

while entry < n_entries:
    next_entry = it.GetNextEntry()

    sm = 0
    susy = 0
    for i in range(int(entry), int(next_entry)):
        tree.GetEntry(i)
        if tree.Label == 0:
            sm += 1
        else:
            susy += 1
        
    total = sm + susy
    print(f"Cluster {n_clusters}: entries={total}, SM={sm} ({100*sm/total:.1f}%), SUSY={susy} ({100*susy/total:.1f}%)")


    n_clusters += 1
    entry = it.Next()

df = ROOT.RDataFrame("CollectionTree", file)

# 
event_info = ["EventNumber", "RunNumber", "dsid", "model"]
particle_info = ["lep_id", "leading_lep"]
kinematics = ["lep1_pt", "lep2_pt", "lep1_eta", "lep2_eta", "lep1_phi", "lep2_phi", "mll", "mt", "mt2_core", "ev_njet", "met_core_sumet"]
jets = ["jet1_pt", "jet1_eta", "jet1_phi", "jet_sumpt"]
model_parameters = ["m1", "m2"]

# define dataset features
training_label = ["Label"]
training_features = kinematics + model_parameters + training_label

# df_signal = df.Filter("Label == 1")
# df_background = df.Filter("Label == 0")

df_sm = ROOT.RDataFrame("CollectionTree", "/home/siliataider/Documents/root/data_loader_bench/SUSY-ATLAS-OpenData-ML/data/SM_SUSY_C1C1_mu_mll-15_MET_both_jets_sm.root")
df_susy = ROOT.RDataFrame("CollectionTree", "/home/siliataider/Documents/root/data_loader_bench/SUSY-ATLAS-OpenData-ML/data/SM_SUSY_C1C1_mu_mll-15_MET_both_jets_susy.root")

dsid_dfs = []
for file in [f for f in pathlib.Path("../data/dsid_files").iterdir() if f.is_file()]:
    dsid_dfs.append(ROOT.RDataFrame("CollectionTree", str(file)))

train, val = ROOT.Experimental.ML.RDataLoader(
    dsid_dfs,
    batch_size = 64,
    batches_in_memory = 999999,
    columns = training_features,
    target = training_label,
    shuffle = True,
    validation_split = 0.3,
)

sm_train, susy_train, total_train = 0, 0, 0
for x, y in train.AsNumpy():
    sm_train += (y == 0).sum()
    susy_train += (y == 1).sum()
    total_train += len(y)

sm_val, susy_val, total_val = 0, 0, 0
for x, y in val.AsNumpy():
    sm_val += (y == 0).sum()
    susy_val += (y == 1).sum()
    total_val += len(y)

print(f"\nTraining:   total={total_train}, SM={sm_train} ({100*sm_train/total_train:.1f}%), SUSY={susy_train} ({100*susy_train/total_train:.1f}%)")
print(f"Validation: total={total_val},  SM={sm_val}  ({100*sm_val/total_val:.1f}%),  SUSY={susy_val}  ({100*susy_val/total_val:.1f}%)")
print(f"SUSY fraction train/val ratio: {(susy_train/total_train) / (susy_val/total_val):.2f}" if susy_val > 0 else "No SUSY in validation!")