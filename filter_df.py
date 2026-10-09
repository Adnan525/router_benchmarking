import pandas as pd
import json
import ast

# df = pd.read_csv("cross-encoder_Qwen-Qwen3-Reranker-4B_val.csv")

# # 1. Parse each string into a dict (one dict per row)
# parsed = df["scores"].apply(ast.literal_eval)

# # 2. Dicts -> DataFrame (keys become columns), then append "_score" to every name
# scores_df = pd.DataFrame(parsed.tolist(), index=df.index).add_suffix("_score")

# # 3. Attach the new columns alongside the existing ones
# df = pd.concat([df, scores_df], axis=1)
# df = df[["expected_service", 
#          'abstract_algebra_score', 
#          'anatomy_score',
#          'astronomy_score']]

# print(df["expected_service"].value_counts())
# df.to_csv("data/k3/validation.csv", index=False)

# # test
# df_test = pd.read_csv("df3_filtered.csv")
# df_test = df_test[["expected_service", 
#                    'abstract_algebra_score', 
#                    'anatomy_score',
#                    'astronomy_score']]
# df_test.to_csv("data/k3/test.csv", index=False)

df = pd.read_csv("data/k5/test.csv")

df = df[["expected_service", 
         'abstract_algebra_score', 
         'anatomy_score',
         'astronomy_score',
         'business_ethics_score',
         'college_computer_science_score']]

df.to_csv("data/k5/test.csv", index=False)