import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap


def draw_umap(data, color_code, legend_handles, title='', n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=color_code)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=color_code)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=color_code, s=100)
    plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(f'./data/psytar/embed_umap_nn={n_neighbors}_comps={n_components}_{metric}-{min_dist}.png')


psytar_embeds = np.loadtxt(f'./data/psytar/psytar_average_doc_vectors.txt', dtype=float)
psytar_intents = []
with open(f'./data/psytar/label', 'r', encoding='utf-8') as intentFile:
    for line in intentFile:
        psytar_intents.append(str(line).strip().replace("\n", ''))
assert len(psytar_embeds) == len(psytar_intents)


sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})

col_map = {'adverse-drug-reaction':'blue',
           'describe-symptom':'green', 
           'drug-indications':'red',
           'sign-symptoms-illness':'black',
           'withdrawal':'pink'}

col_res = [col_map[c] for c in list(psytar_intents)]

handles = [mpatches.Patch(color=v, label=k) for k,v in col_map.items()]


for nn in [100]:#[5,10,20,50,100,150,200,250,500,1000]:
    draw_umap(data=psytar_embeds, 
            color_code=col_res,
            legend_handles=handles,
            title='UMAP of token-averaged bioreddit GloVe embeddings in Psytar Dataset lines',
            n_neighbors=nn,
            min_dist = 0.3,
            n_components=2,
            metric='cosine'
            )