import ast
import pandas as pd
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

subjects = ['abstract_algebra', 'anatomy', 'astronomy']

colors = dict(zip(subjects + ['general'], 
                  ['red', 'green', 'blue', 'black']))

df = pd.read_csv("data/cross-encoder_Qwen-Qwen3-Reranker-4B_val.csv")
df = df[['expected_service', 'scores', 'zscores']]

score = pd.DataFrame(
            df['scores'].apply(ast.literal_eval).tolist()
        )[subjects].add_suffix('_score')

zscore = pd.DataFrame(
            df['zscores'].apply(ast.literal_eval).tolist()
        )[subjects].add_suffix('_zscore')

sm = pd.DataFrame(softmax(score.to_numpy(), axis=1),
                 columns=[f'{s}_softmax' for s in subjects])

df = pd.concat([df['expected_service'], score, zscore, sm], axis=1)

fig = plt.figure(figsize=(24, 10))
for i, kind in enumerate(['score', 'zscore', 'softmax']):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    for service, g in df.groupby('expected_service'):
        ax.scatter(*(g[f'{s}_{kind}'] for s in subjects),
                   c=colors[service], label=service, s=30, alpha=.8)
    ax.set_xlabel(subjects[0], fontsize=20)
    ax.set_ylabel(subjects[1], fontsize=20)
    ax.set_zlabel(subjects[2], fontsize=20)
    ax.set_title(kind, fontsize=24)
    ax.view_init(25, 45)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=24, markerscale=2)

plt.tight_layout(rect=(0.03, 0.06, 1, 1))  # leave room for the z-label and legend
plt.savefig('scores_3d.png', dpi=140)
