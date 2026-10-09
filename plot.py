import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = pd.read_csv('df3_filtered.csv')

subjects = ['abstract_algebra', 'anatomy', 'astronomy']
colors = dict(zip(subjects + ['general'], ['red', 'green', 'blue', 'black']))

fig = plt.figure(figsize=(24, 8))
for i, kind in enumerate(['score', 'zscore', 'softmax']):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    for s, g in d.groupby('expected_service'):
        general = s == 'general'
        ax.scatter(*(g[f'{x}_{kind}'] for x in subjects), c=colors[s], label=s,
                   s=28 if general else 12, alpha=.9 if general else .35,
                   marker='^' if general else 'o')
    ax.set(xlabel=subjects[0], ylabel=subjects[1], zlabel=subjects[2],
           title=f'{kind}s by expected service')
    ax.view_init(25, 45)
    if i == 0: ax.legend()

plt.tight_layout()
plt.savefig('scores_3d.png', dpi=140)
