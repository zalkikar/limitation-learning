from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from config import col_map
import matplotlib.pyplot as plt

psytar_embeds = np.loadtxt(f'./data/psytar/psytar_average_doc_vectors.txt', dtype=float)
psytar_intents = []
with open(f'./data/psytar/label', 'r', encoding='utf-8') as intentFile:
    for line in intentFile:
        psytar_intents.append(str(line).strip().replace("\n", ''))
assert len(psytar_embeds) == len(psytar_intents)


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(psytar_embeds)
df = pd.DataFrame(psytar_intents, columns =['target'])
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2', 'pc3'])

finalDf = pd.concat([principalDf, df['target']], axis = 1)

fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_zlabel('Principal Component 3', fontsize = 15)
ax = fig.add_subplot(111, projection='3d')

ax.set_title('PCA of token-averaged bioreddit GloVe embeddings in Psytar Dataset lines', fontsize = 13)
for target, color in col_map.items():
    indicesToKeep = finalDf['target'] == target
    if target == 'adverse-drug-reaction' or target == 'describe-symptom':
        a = 0.2
    else:
        a = 1.
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
               , finalDf.loc[indicesToKeep, 'pc2']
               , finalDf.loc[indicesToKeep, 'pc3']
               , c = color
               , s = 50
               , alpha =a)
ax.legend(list(col_map.keys()))
for angle1 in [0,225]:
    for angle2 in [0,45,90,180,225,270,315,360]:
        ax.view_init(angle1, angle2)
        ax.grid()
        plt.tight_layout()
        plt.savefig(f'./data/psytar/embed_pca_nn=3_rotation={angle1}by{angle2}.png')