from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,f1_score,accuracy_score
from tqdm import tqdm 
import scipy
import joblib
from collections import Counter
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random 
from scipy.stats import percentileofscore

np.random.seed(10086)
random.seed(10086)

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
print(current_dir)


def reverse_index_map(before_lst,after_lst):
    '''
    before_lst: orginal index lst [1,2,3,4,5,6]
    after_lst : index lst shuffled [2,3,1,6,5,4]
    '''
    map_dict = {}
    for idx in range(len(after_lst)):
        map_dict[after_lst[idx]] = idx
    return_index = [map_dict[x] for x in before_lst]
    return return_index

def compute_aupr(y_true,y_predicted):
    precision, recall, thresholds = precision_recall_curve(y_true, y_predicted)
    return auc(recall,precision)

def strict_nfold(dat,seed=1016):
    fold = 5
    dat.reset_index(inplace=True,drop=True)
    original_index = dat.index
    true_y,pred_y,index_order = [],[],[]
    aucTotal,esm_auctotal = 0,0
    auprTotal,esm_auprtotal = 0,0
    corTotal,esm_corTotal = 0,0
    source_genenames =list(Counter(dat['gene_name'].values).keys())
    source_genenames = np.array(source_genenames)
    np.random.shuffle(source_genenames)
    kf = KFold(n_splits=fold,shuffle=True,random_state=seed)
    for _,(train_genename_idx,val_genename_idx) in enumerate(kf.split(source_genenames)):
        train_genenames, val_genenames = \
            source_genenames[train_genename_idx],source_genenames[val_genename_idx]  
        print('training testing source size',len(train_genenames),len(val_genenames))
        train_dat = pd.concat([dat[dat['gene_name']==s] for s in train_genenames],
                              axis=0,ignore_index=True)
        train_x = train_dat.drop(['label','gene_name'],axis=1).values
        train_y = train_dat['label'].values
        test_dat = pd.concat([dat[dat['gene_name']==s] for s in val_genenames],axis=0)
        test_x = test_dat.drop(['label','gene_name'],axis=1).values
        test_y = test_dat['label'].values
        model = RandomForestClassifier(
            n_estimators=200,class_weight='balanced',max_depth=15,
            random_state=seed)
        model.fit(train_x,train_y)
        predictedy = model.predict_proba(test_x)[:,1]
        p_train_y = model.predict_proba(train_x)[:,1]
        print('roc_auc:',roc_auc_score(test_y, predictedy))
        print('aupr score:',compute_aupr(test_y,predictedy))
        print('training AUC:',roc_auc_score(train_y, p_train_y))
        aucTotal+=roc_auc_score(test_y, predictedy)
        auprTotal+=compute_aupr(test_y, predictedy)
        corTotal+=scipy.stats.pearsonr(test_y, predictedy)[0]
        pred_y += list(np.squeeze(predictedy))
        true_y += list(test_y)
        index_order += list(test_dat.index)
        del model
    print(f'aupr average:{auprTotal/fold},auc average{aucTotal/fold}')
    print(f'esm aupr average:{esm_auprtotal/fold},auc average{esm_auctotal/fold}')
    order = reverse_index_map(original_index,index_order)
    pred_y = np.array(pred_y)[order]
    print(len(true_y))
    true_y = np.array(true_y)[order]
    return pred_y,true_y

clinvar_data_merged_esm1b_psmutpred = pd.read_csv(
    'data/dataset/psmutpred_ps_esm_trainandCV_set_dropdupli_addplddt.tsv',
    sep='\t')
cdmep = clinvar_data_merged_esm1b_psmutpred.copy()

'''
generate labels and convert string to float
'''
def get_label(sig):
    if sig in ['Pathogenic','Likely pathogenic']:
        return 1
    elif sig in ['Benign','Likely benign']:
        return 0
    else:
        print(sig)
  
cdmep['label'] = cdmep['clinvar_significance'].apply(lambda x:get_label(x))
cdmep['pfam'] = cdmep['in_pfam'].apply(lambda x:1 if x=='pfam' else 0)

'''
add one-hot
'''
RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_to_index = {aa: index for index, aa in enumerate(RESIDUES)}

# 创建一个函数来生成one-hot编码
def create_one_hot(row):
    one_hot_wt = np.zeros(len(RESIDUES))
    one_hot_mt = np.zeros(len(RESIDUES))
    one_hot_wt[aa_to_index[row['wt_aa']]] = 1
    one_hot_mt[aa_to_index[row['mt_aa']]] = 1
    one_hot = np.concatenate([one_hot_wt, one_hot_mt])
    return one_hot
cdmep_one_hot = pd.DataFrame(
    np.vstack(cdmep.apply(create_one_hot, axis=1)), 
    columns=[f'wt_{aa}' for aa in RESIDUES] + [f'mt_{aa}' for aa in RESIDUES])
cdmep = pd.concat([cdmep,cdmep_one_hot],axis=1)

'''
filter columns
'''

cdmep.reset_index(drop=True,inplace=True)


cdmep['pLDDT'].fillna(cdmep['pLDDT'].median(),inplace=True)
cdmep.drop(['uniprot_entry','wt_aa','mt_aa','position','PhaSeScore','PhaSeRank', 
                'cat_aa_wt', 'cat_aa_diff', 'catgra_seq', 
                'clinvar_significance','in_pfam',
                ],axis=1,inplace=True
               )
cdmep.dropna(inplace=True,)
cdmep_original = cdmep.copy()
cdmep_plddt_lst = np.array(cdmep['pLDDT'].values)
cdmep.drop('pLDDT',axis=1,inplace=True
               ) # avoid it from taken by the model
         
'''
nfold
'''

py,ty = strict_nfold(cdmep)
print('total auc:',roc_auc_score(ty,py))
print('total aupr:',compute_aupr(ty,py))

print(cdmep_original.shape,cdmep.shape,len(ty),len(py))
cdmep_original['predicted_by_model'] = py 
cdmep_original['label'] = ty
cdmep_original['pLDDT'] = cdmep_plddt_lst
cdmep_original.to_csv('clinvar_esm1b_ps_pred.tsv',sep='\t',index=False)


'''
nfold duplicates 
'''
# cdmep_to_save = cdmep.copy()
# print(cdmep.shape)
# for idx in range(1):
#     np.random.seed(idx*19)
#     py,ty = strict_nfold(cdmep)

#     print('total auc:',roc_auc_score(ty,py))
#     print('total aupr:',compute_aupr(ty,py))

#     cdmep_to_save[f'predicted_by_model_{idx}'] = py 
#     cdmep_to_save[f'label_of_eval_{idx}'] = ty
#     cdmep_to_save.to_csv('data/predicted_results/clinvar_esm1b_ps_pred.tsv',sep='\t',index=False)
'''
evaluation
'''
cdmep = pd.read_csv('clinvar_esm1b_ps_pred.tsv',sep='\t')
print(cdmep.head(10),cdmep.shape)

cdmep_idr = cdmep[cdmep['pfam']==0]
cdmep_idr_disorder = cdmep_idr[cdmep_idr['iupred_score']>=0.5]
cdmep_domain = cdmep[cdmep['pfam']==1]

cdmep_vrylow_plddt = cdmep[cdmep['pLDDT']<30]
cdmep_low_plddt = cdmep[(cdmep['pLDDT']>=30)&(cdmep['pLDDT']<50)]
cdmep_medium_plddt = cdmep[(cdmep['pLDDT']>=50)&(cdmep['pLDDT']<70)]
cdmep_high_plddt = cdmep[cdmep['pLDDT']>=70]
cdmep_idr_low_plddt = cdmep[(cdmep['pLDDT']<50)&(cdmep['pfam']==0)]
cdmep_idr_high_plddt = cdmep[(cdmep['pLDDT']>=70)&(cdmep['pfam']==0)]

print('ESM-1b original')
print('################################')
print('domain class weight',Counter(cdmep_domain['label'].values))
print('domain auc',roc_auc_score(cdmep_domain['label'].values,
                    -cdmep_domain['esm_score'].values))
print('domain aupr',compute_aupr(cdmep_domain['label'].values,
                    -cdmep_domain['esm_score'].values))

print('idr class weight',Counter(cdmep_idr['label'].values))
print('idr auc',roc_auc_score(cdmep_idr['label'].values,
                    -cdmep_idr['esm_score'].values))
print('idr aupr',compute_aupr(cdmep_idr['label'].values,
                    -cdmep_idr['esm_score'].values))

print('IDR low plddt class weight',
      Counter(cdmep_idr_low_plddt['label'].values))
print('IDR low plddt auc',roc_auc_score(cdmep_idr_low_plddt['label'].values,
                    -cdmep_idr_low_plddt['esm_score'].values))
print('IDR low plddt aupr',compute_aupr(cdmep_idr_low_plddt['label'].values,
                    -cdmep_idr_low_plddt['esm_score'].values))

print('IDR high plddt class weight',
      Counter(cdmep_idr_high_plddt['label'].values))
print('IDR high plddt auc',roc_auc_score(cdmep_idr_high_plddt['label'].values,
                    -cdmep_idr_high_plddt['esm_score'].values))
print('IDR high plddt aupr',compute_aupr(cdmep_idr_high_plddt['label'].values,
                    -cdmep_idr_high_plddt['esm_score'].values))


print('ESM-1b+PS')
print('################################')
print('domain class weight',Counter(cdmep_domain['label'].values))
print('domain auc',roc_auc_score(cdmep_domain['label'].values,
                    cdmep_domain['predicted_by_model'].values))
print('domain aupr',compute_aupr(cdmep_domain['label'].values,
                    cdmep_domain['predicted_by_model'].values))

print('idr class weight',Counter(cdmep_idr['label'].values))
print('idr auc',roc_auc_score(cdmep_idr['label'].values,
                    cdmep_idr['predicted_by_model'].values))
print('idr aupr',compute_aupr(cdmep_idr['label'].values,
                    cdmep_idr['predicted_by_model'].values))
print('IDR low plddt class weight',
      Counter(cdmep_idr_low_plddt['label'].values))
print('IDR low plddt auc',roc_auc_score(cdmep_idr_low_plddt['label'].values,
                    cdmep_idr_low_plddt['predicted_by_model'].values))
print('IDR low plddt aupr',compute_aupr(cdmep_idr_low_plddt['label'].values,
                    cdmep_idr_low_plddt['predicted_by_model'].values))

print('IDR high plddt class weight',
      Counter(cdmep_idr_high_plddt['label'].values))
print('IDR high plddt auc',roc_auc_score(cdmep_idr_high_plddt['label'].values,
                cdmep_idr_high_plddt['predicted_by_model'].values))
print('IDR high plddt aupr',compute_aupr(cdmep_idr_high_plddt['label'].values,
                    cdmep_idr_high_plddt['predicted_by_model'].values))
