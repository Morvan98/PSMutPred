print('---importing models---')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score,accuracy_score,auc,precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats

from modules import *


import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def compute_aupr(y_true,y_predicted):
    precision, recall, _ = precision_recall_curve(y_true, y_predicted)
    return auc(recall,precision)

def compute_class_weight(lst):
    cw1,cw0 = 1,1
    lst = list(np.squeeze(lst))
    count1 = lst.count(1)
    count0 = lst.count(0)
    if count1 > count0:
        cw0 = count1/count0    
    else:
        cw1 = count0/count1
    return {1:cw1,0:cw0}

def reverse_index_map(before_lst, after_lst):
    '''
    example:
    before_lst: original index list [1,2,3,4,5,6]
    after_lst : index list shuffled [2,3,1,6,5,4]
    '''
    map_dict = {value: idx for idx, value in enumerate(after_lst)}
    return [map_dict[x] for x in before_lst]

def score_filter(score):
    return format(score,'.3f')

RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
def create_one_hot(row):
    RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_to_index = {aa: index for index, aa in enumerate(RESIDUES)}
    one_hot_wt = np.zeros(len(RESIDUES))
    one_hot_mt = np.zeros(len(RESIDUES))
    one_hot_wt[aa_to_index[row['wt_aa']]] = 1
    one_hot_mt[aa_to_index[row['mt_aa']]] = 1
    one_hot = np.concatenate([one_hot_wt, one_hot_mt])
    return one_hot

def strict_nfold(dat):
    fold = 3
    source_names = list(Counter(list(dat['GeneName'].values)).keys())
    validation_source_num = len(source_names)//fold
    true_y,pred_y = [],[]
    aucTotal, corTotal = 0,0
    source_names = np.array(source_names)
    np.random.shuffle(source_names)
    source_names = list(source_names)
    for i in range(fold):
        test_source = source_names[i*validation_source_num:(i+1)*validation_source_num]
        if i == fold - 1: # Avoid incomplete processing of the last fold
            test_source = source_names[i*validation_source_num:]
        train_source = [source for source in source_names if source not in test_source]
        train_dat = pd.concat([dat[dat['GeneName']==s] for s in train_source],axis=0)
        train_X = train_dat.drop(['PathoLabel','GeneName'],axis=1).values
        train_Y = train_dat['PathoLabel'].values
        test_dat = pd.concat([dat[dat['GeneName']==s] for s in test_source],axis=0)
        test_X = test_dat.drop(['PathoLabel','GeneName'],axis=1).values
        test_Y = test_dat['PathoLabel'].values
        scaler = MinMaxScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        model = RandomForestClassifier(n_estimators=200,class_weight='balanced',max_depth=15)
        model.fit(train_X,train_Y)
        predictedY = model.predict_proba(test_X)[:,1]
        auc_score = roc_auc_score(test_Y, predictedY)
        pearson_score = scipy.stats.pearsonr(test_Y, predictedY)[0]
        print(f'---fold AUROC:{score_filter(auc_score)}; fold Pearson:{score_filter(pearson_score)}')
        aucTotal+=roc_auc_score(test_Y, predictedY)
        corTotal+=scipy.stats.pearsonr(test_Y, predictedY)[0]
        pred_y += list(np.squeeze(predictedY))
        true_y += list(test_Y)
        del model
    pearson_avg,auc_avg = corTotal/fold,aucTotal/fold
    auc_total,pearson_total = roc_auc_score(true_y, pred_y),scipy.stats.pearsonr(true_y, pred_y)[0]
    print(f'---average AUROC:{score_filter(auc_avg)}; average Pearson:{score_filter(pearson_avg)}')
    print(f'---total AUROC:{score_filter(auc_total)}; total Pearson:{score_filter(pearson_total)}')
    return aucTotal/fold,pred_y,true_y
print('---preparating dataset---')
cv_data = pd.read_csv('data/dataset/patho_cv_data.tsv',sep='\t')
independent_test_data = pd.read_csv('data/dataset/patho_test_data.tsv',sep='\t') # removed variants in the same amino acid position seen in the cross-validation set to avoid data leakage

cv_data.drop(['wt_aa','mt_aa',],axis=1,inplace=True)
cv_data.dropna(inplace=True,axis=0)
cv_data.reset_index(inplace=True,drop=True)

'''
Evaluation on the cross-validation set using a blocked 3-fold cross-validation 
where we strictly separate variants from the same gene into the same group.
'''
print('---performance on cross-validation set')
print('---EVE performance')
eve_auc,eve_pearson = roc_auc_score(cv_data['PathoLabel'].values,cv_data['EVE_scores'].values),scipy.stats.pearsonr(cv_data['PathoLabel'].values,cv_data['EVE_scores'].values)[0]
print(f'---EVE AUROC:{score_filter(eve_auc)}; EVE Pearson:{score_filter(eve_pearson)}')

print('---Performance when combining PS features')

print(f'----3 fold cross-validation performance eval')
_,_,_ = strict_nfold(cv_data)



'''
Evaluation on independent test set
To construct an independent test dataset, we screened new variants (ClinVar50 annotations up to 2022.12) 
that were not used to evaluate the EVE model and removed variants in the same amino acid position seen in 
the cross-validation set to avoid data leakage.
'''


independent_test_data.drop(['wt_aa','mt_aa',],axis=1,inplace=True)
independent_test_data.dropna(inplace=True,axis=0)
independent_test_data.reset_index(inplace=True,drop=True)
test_data_pfaminfo = list(independent_test_data['pfam'].values)

idr_samples = [x for x,item in enumerate(test_data_pfaminfo) if item==0]
domain_samples = [x for x,item in enumerate(test_data_pfaminfo) if item==1]

print('---performance on independent test set')
print('---EVE performance')
auc_ = roc_auc_score(independent_test_data['PathoLabel'].values, independent_test_data['EVE_scores'].values)
aupr_ = compute_aupr(independent_test_data['PathoLabel'].values, independent_test_data['EVE_scores'].values)
pearson_ = scipy.stats.pearsonr(independent_test_data['PathoLabel'].values, independent_test_data['EVE_scores'].values)[0]
n_variants = len(independent_test_data['PathoLabel'].values)
print(f'---Total AUROC:{score_filter(auc_)}; total AUPR:{score_filter(aupr_)}; total Pearson:{score_filter(pearson_)}; {n_variants} variants')


auc_ = roc_auc_score(independent_test_data['PathoLabel'].values[idr_samples], independent_test_data['EVE_scores'].values[idr_samples])
aupr_ = compute_aupr(independent_test_data['PathoLabel'].values[idr_samples], independent_test_data['EVE_scores'].values[idr_samples])
pearson_ = scipy.stats.pearsonr(independent_test_data['PathoLabel'].values[idr_samples], independent_test_data['EVE_scores'].values[idr_samples])[0]
n_variants = len(independent_test_data['PathoLabel'].values[idr_samples])
print(f'---IDR variants AUROC:{score_filter(auc_)}; total AUPR:{score_filter(aupr_)}; total Pearson:{score_filter(pearson_)}; {n_variants} variants')

auc_ = roc_auc_score(independent_test_data['PathoLabel'].values[domain_samples], independent_test_data['EVE_scores'].values[domain_samples])
aupr_ = compute_aupr(independent_test_data['PathoLabel'].values[domain_samples], independent_test_data['EVE_scores'].values[domain_samples])
pearson_ = scipy.stats.pearsonr(independent_test_data['PathoLabel'].values[domain_samples], independent_test_data['EVE_scores'].values[domain_samples])[0]
n_variants = len(independent_test_data['PathoLabel'].values[domain_samples])
print(f'---Domain variants AUROC:{score_filter(auc_)}; total AUPR:{score_filter(aupr_)}; total Pearson:{score_filter(pearson_)}; {n_variants} variants')


train_X = cv_data.drop(['PathoLabel','GeneName'],axis=1).values
train_Y = cv_data['PathoLabel'].values
print(train_X.shape)

print('---Performance when combining PS features using Random Forest model')
test_X = independent_test_data.drop(['PathoLabel','GeneName'],axis=1).values
test_Y = independent_test_data['PathoLabel'].values
model = RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=111)
model.fit(train_X,train_Y)
predictedY = model.predict_proba(test_X)[:,1] 

auc_ = roc_auc_score(test_Y, predictedY)
aupr_ = compute_aupr(test_Y, predictedY)
pearson_ = scipy.stats.pearsonr(test_Y, predictedY)[0]
n_variants = len(np.array(test_Y))
print(f'---Total AUROC:{score_filter(auc_)}; total AUPR:{score_filter(aupr_)}; total Pearson:{score_filter(pearson_)}; {n_variants} variants')

auc_ = roc_auc_score(np.array(test_Y)[idr_samples], predictedY[idr_samples])
aupr_ = compute_aupr(np.array(test_Y)[idr_samples], predictedY[idr_samples])
pearson_ = scipy.stats.pearsonr(np.array(test_Y)[idr_samples], predictedY[idr_samples])[0]
n_variants = len(np.array(test_Y)[idr_samples])
print(f'---IDR variants AUROC:{score_filter(auc_)}; total AUPR:{score_filter(aupr_)}; total Pearson:{score_filter(pearson_)}; {n_variants} variants')

auc_ = roc_auc_score(np.array(test_Y)[domain_samples], predictedY[domain_samples])
aupr_ = compute_aupr(np.array(test_Y)[domain_samples], predictedY[domain_samples])
pearson_ = scipy.stats.pearsonr(np.array(test_Y)[domain_samples], predictedY[domain_samples])[0]
n_variants = len(np.array(test_Y)[domain_samples])
print(f'---Domain variants AUROC:{score_filter(auc_)}; total AUPR:{score_filter(aupr_)}; total Pearson:{score_filter(pearson_)}; {n_variants} variants')



