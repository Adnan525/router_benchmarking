# import pandas as pd
# from scipy.special import softmax

# df = pd.read_csv('df3_filtered.csv')

# score_cols = ['abstract_algebra_score', 'anatomy_score', 'astronomy_score']
# softmax_cols = [c.replace('_score', '_softmax') for c in score_cols]

# df[softmax_cols] = softmax(df[score_cols].to_numpy(dtype=float), axis=1)

# df = df[['expected_service', 'predicted_label'] + score_cols +
#         ['abstract_algebra_zscore', 'anatomy_zscore', 'astronomy_zscore'] +
#         softmax_cols]

# df.to_csv('df3_filtered.csv', index=False)
# print(df.columns)

from GatePredictor import GatePredictor

VAL_PATHS = ["data/k3/validation.csv", 
            "data/k5/validation.csv",
            "data/k8/validation.csv",
            "data/k10/validation.csv"
            ]

DATA_PATHS =["data/k3/test.csv", 
            "data/k5/test.csv",
            "data/k8/test.csv",
            "data/k10/test.csv"
            ]

CUTOFFS = [1.5, 1.75, 2.0, 2.25, 2.5]
# CUTOFFS = [2.5]

for val_path, data_path in zip(VAL_PATHS, DATA_PATHS):

    gate = GatePredictor(val_path, data_path, CUTOFFS)
    gate_accuracy, real_accuracy = gate.accuracy()

    print(f"K = {gate.calibrator.k}")
    print(f"tau = {gate.tau:.3f}, cut-off = {gate.cutoff:.2f}")
    print(f"general vs specialist accuracy : {gate_accuracy:.4f}")
    print(f"real routing accuracy          : {real_accuracy:.4f}")
    print("=" * 50)