import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

csv_file = '/home/siliataider/Documents/root/data_loader_bench/SUSY-ATLAS-OpenData-ML/data/SM_SUSY_C1C1_mu_mll-15_MET_both_jets.csv'
parquet_file = '/home/siliataider/Documents/root/data_loader_bench/SUSY-ATLAS-OpenData-ML/data/SM_SUSY_C1C1_mu_mll-15_MET_both_jets.parquet'
chunksize = 3_000_000

feature_cols = None

csv_stream = pd.read_csv(csv_file, chunksize=chunksize)

parquet_writer = None

event_info = ["EventNumber", "RunNumber", "dsid", "model"]
particle_info = ["lep_id", "leading_lep"]
kinematics = ["lep1_pt", "lep2_pt", "lep1_eta", "lep2_eta", "lep1_phi", "lep2_phi", "mll", "mt", "mt2_core", "ev_njet", "met_core_sumet"]
jets = ["jet1_pt", "jet1_eta", "jet1_phi", "jet_sumpt"]
model_parameters = ["m1", "m2"]

# define dataset features
training_label = ["Label"]
training_features = kinematics + model_parameters + training_label


for i, chunk in enumerate(csv_stream):
    print("Chunk", i)

    feature_cols = [c for c in training_features if c != "Label"]

    features = chunk[feature_cols].to_numpy(dtype=np.float32)
    labels = chunk["Label"].to_numpy(dtype=np.float32)

    chunk["features"] = list(features)
    chunk["Label"] = labels

    table = pa.Table.from_pandas(
        chunk[["features", "Label"]],
        preserve_index=False
    )

    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(
            parquet_file,
            table.schema,
        )

    parquet_writer.write_table(table)

parquet_writer.close()