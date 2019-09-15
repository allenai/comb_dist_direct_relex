import json
from sklearn.metrics import precision_recall_curve
from scipy.interpolate import spline
import matplotlib.pyplot as plt

with open('scripts/PR_curves.json') as f:
    x = json.load(f)

plt.step(x['belagy_et_al_best'][0], x['belagy_et_al_best'][1], where='post')
plt.step(x['belagy_et_al_baseline'][0], x['belagy_et_al_baseline'][1], where='post')
plt.step(x['reside'][0], x['reside'][1], where='post')
plt.step(x['lin_et_al'][0], x['lin_et_al'][1], where='post')
plt.grid( linestyle='dashed', linewidth=0.5)
plt.legend(['This work', 
            'Baseline',
            'RESIDE (Vashishth et al., 2018)',
            'PCNN+ATT (Lin et al., 2016)',
            ])
plt.xlabel('recall')
plt.ylabel('precision')
plt.ylim([0.4, 1])
plt.xlim([0, 0.4])
