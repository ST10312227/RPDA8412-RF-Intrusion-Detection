import json, os, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score)
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D=os.path.join(BASE,'data'); R=os.path.join(BASE,'analysis','results')
RS=42
def bm(y,p,pr):
    cm=confusion_matrix(y,p); tn,fp,fn,tp=cm.ravel()
    return {"accuracy":float(accuracy_score(y,p)),"precision":float(precision_score(y,p,zero_division=0)),
            "recall":float(recall_score(y,p,zero_division=0)),"f1":float(f1_score(y,p,zero_division=0)),
            "roc_auc":float(roc_auc_score(y,pr)),"pr_auc":float(average_precision_score(y,pr)),
            "fpr":float(fp/(fp+tn)),"fnr":float(fn/(fn+tp)),"specificity":float(tn/(tn+fp)),
            "balanced_accuracy":float(balanced_accuracy_score(y,p)),"mcc":float(matthews_corrcoef(y,p)),
            "kappa":float(cohen_kappa_score(y,p)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"n":int(len(y))}
Xtr=pd.read_pickle(os.path.join(D,'unsw_X_train.pkl')).astype(np.float32)
Xte=pd.read_pickle(os.path.join(D,'unsw_X_test.pkl')).astype(np.float32)
ytr=np.load(os.path.join(D,'unsw_y_train_bin.npy')); yte=np.load(os.path.join(D,'unsw_y_test_bin.npy'))
rng=np.random.RandomState(RS); idx=rng.choice(len(Xtr),size=60000,replace=False)
Xs,ys=Xtr.values[idx],ytr[idx]
out=json.load(open(os.path.join(R,'unsw_imbalance.json')))
# subset baseline for a fair comparison
rf0=RandomForestClassifier(n_estimators=200,random_state=RS,n_jobs=2,max_features='sqrt')
rf0.fit(Xs,ys)
out['smote_subset_baseline']=bm(yte,rf0.predict(Xte.values),rf0.predict_proba(Xte.values)[:,1])
print('subset baseline', {k:round(v,4) for k,v in out['smote_subset_baseline'].items() if isinstance(v,float)})
t=time.time()
Xres,yres=SMOTE(random_state=RS,sampling_strategy='not majority',k_neighbors=5).fit_resample(Xs,ys)
out['smote_train_size']=int(len(Xres))
out['smote_class_counts']={int(k):int(v) for k,v in zip(*np.unique(yres,return_counts=True))}
rf=RandomForestClassifier(n_estimators=200,random_state=RS,n_jobs=2,max_features='sqrt')
rf.fit(Xres,yres)
out['smote_train_time_s']=time.time()-t
out['smote']=bm(yte,rf.predict(Xte.values),rf.predict_proba(Xte.values)[:,1])
json.dump(out,open(os.path.join(R,'unsw_imbalance.json'),'w'),indent=2,default=str)
print('smote', {k:round(v,4) for k,v in out['smote'].items() if isinstance(v,float)})
