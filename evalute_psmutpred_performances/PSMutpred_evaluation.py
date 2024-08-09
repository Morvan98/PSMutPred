from modules import *
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mannwhitneyu,ks_2samp

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def compute_aupr(y_true,y_predicted):
    precision, recall, _ = precision_recall_curve(y_true, y_predicted)
    return auc(recall,precision)

def discriminative_power(y_true,y_predicted):
    positive_prediction,negative_prediction = [],[]
    for idx,y in enumerate(y_true):
        if y==1:
            positive_prediction.append(y_predicted[idx])
        else:
            negative_prediction.append(y_predicted[idx])
    print('discriminative power between positive and negative samples: ')
    print(mannwhitneyu(positive_prediction,negative_prediction))
    print(ks_2samp(positive_prediction,negative_prediction))
    
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

'''
existing existing ps model's performances using all the collated data
'''
existing_ps_predictors_score_diff_values = pd.read_csv(
    'data/dataset/existing_ps_predictors_score_diff.tsv',sep='\t')
'''
evaluate existing phase separation predictors on 'impact' mutations
'''
exist_model_perf_ip = existing_ps_predictors_score_diff_values.copy()
label_ip = exist_model_perf_ip['label'].values 
exist_model_perf_ip['label'] = np.where(label_ip!=0,1,0)
'''
evaluate discriminative power on mutation that impact ps
'''
print('---analyzing results of existing ps models---')
print('---evaluating existing ps models on ip task using descriminative power---')
print('---DeePhase---')
discriminative_power(exist_model_perf_ip['label'].values,abs(exist_model_perf_ip['deephase_diff'].values))
print('---PSAP---')
discriminative_power(exist_model_perf_ip['label'].values,abs(exist_model_perf_ip['psap_diff'].values))
print('---FuzDrop---')
discriminative_power(exist_model_perf_ip['label'].values,abs(exist_model_perf_ip['fuzdrop_residue_diff'].values))
print('---catGRANULE---')
discriminative_power(exist_model_perf_ip['label'].values,abs(exist_model_perf_ip['catgra_diff'].values))
print('---PScore---')
discriminative_power(exist_model_perf_ip['label'].values,abs(exist_model_perf_ip['pscore_diff'].values))
'''
evaluate using roc_auc scores on mutation that impact ps
'''
print('---evaluating existing ps models on ip task using auroc---')
print('---DeePhase---')
print(roc_auc_score(exist_model_perf_ip['label'].values,abs(exist_model_perf_ip['deephase_diff'].values)))
print('---PSAP---')
print(roc_auc_score(exist_model_perf_ip['label'].values,abs(exist_model_perf_ip['psap_diff'].values)))
print('---FuzDrop---')
print(roc_auc_score(exist_model_perf_ip['label'].values,abs(exist_model_perf_ip['fuzdrop_residue_diff'].values)))
print('---catGRANULE---')
print(roc_auc_score(exist_model_perf_ip['label'].values,abs(exist_model_perf_ip['catgra_diff'].values)))
print('---PScore---')
print(roc_auc_score(exist_model_perf_ip['label'].values,abs(exist_model_perf_ip['pscore_diff'].values)))

'''
evaluate existing phase separation predictors on 'strengthen/weaken' mutations
'''
exist_model_perf_sp = existing_ps_predictors_score_diff_values.copy()
exist_model_perf_sp = exist_model_perf_sp[exist_model_perf_sp['label']!=0] ### consider 'impact mutations' (strengthen/weaken) only
label_sp = exist_model_perf_sp['label'].values 
exist_model_perf_sp['label'] = np.where(label_sp==2,1,0)
'''
evaluate discriminative power on 'strengthen/weaken' mutations
'''
print('---evaluating existing ps models on ip task using descriminative power---')
print('---DeePhase---')
discriminative_power(exist_model_perf_sp['label'].values,exist_model_perf_sp['deephase_diff'].values)
print('---PSAP---')
discriminative_power(exist_model_perf_sp['label'].values,exist_model_perf_sp['psap_diff'].values)
print('---FuzDrop---')
discriminative_power(exist_model_perf_sp['label'].values,exist_model_perf_sp['fuzdrop_residue_diff'].values)
print('---catGRANULE---')
discriminative_power(exist_model_perf_sp['label'].values,exist_model_perf_sp['catgra_diff'].values)
print('---PScore---')
discriminative_power(exist_model_perf_sp['label'].values,exist_model_perf_sp['pscore_diff'].values)
'''
evaluate using roc_auc scores on mutation that impact ps
'''
print('---evaluating existing ps models on ip task using auroc---')
print('---DeePhase---')
print(roc_auc_score(exist_model_perf_sp['label'].values,exist_model_perf_sp['deephase_diff'].values))
print('---PSAP---')
print(roc_auc_score(exist_model_perf_sp['label'].values,exist_model_perf_sp['psap_diff'].values))
print('---FuzDrop---')
print(roc_auc_score(exist_model_perf_sp['label'].values,exist_model_perf_sp['fuzdrop_residue_diff'].values))
print('---catGRANULE---')
print(roc_auc_score(exist_model_perf_sp['label'].values,exist_model_perf_sp['catgra_diff'].values))
print('---PScore---')
print(roc_auc_score(exist_model_perf_sp['label'].values,exist_model_perf_sp['pscore_diff'].values))


print('---analyzing results of PSMutPred models---')

'''
47 human proteins for cross-validation dataset 
'''
cross_validation_proteins = ['STING1', 'HNRNPH1', 'USP42', 'YAP1', 'DDX3X', 'UBQLN2', 
               'HTT', 'PRNP', 'ANXA11', 'PTPN11', 'CGAS', 'PLK4', 'LGALS3', 
               'AKAP8', 'CBX5', 'TP53', 'HNRNPDL', 'DDX21', 'GIT1', 'LAT', 
               'FUS', 'MAPT', 'TOPBP1', 'HNRNPA1', 'CAPRIN1', 'EZH2', 'TAF15', 
               'EFHD2', 'TFAM', 'KDM6A', 'TARDBP', 'LEMD2', 'SARM1', 'CIDEC',
                'FBL', 'CD2AP', 'RAPSN', 'TIA1', 'CHAF1A', 'SPOP', 'HNRNPA2B1', 
                'MECP2', 'HSF1', 'G3BP1', 'PTPN6', 'YTHDC1', 'L1RE1']
'''
23 non-human proteins (experimental sequence were non-human) for independent test set
'''
independent_test_set_proteins = ['znf207', 'Dvl2', 'Rv1747', 'NCAP', 'MCL19.12', 'Dronpa', 'Dhh1', 
                'pon', 'SCAF', 'pgl-1', 'Shank3', 'Ape1', 'B1MUE8', 
                'Cbx2', 'Syngap1', 'SARS2-NP', 'TMF', 
                'pros', 'ParB', 'CHLRE_10g436550v5', 'Numb', 'sop-2', 'YFR016C',]


#######################################################

##########################################################################################
'''
PSMutPred-IP leave-one-source-out cross-validation (IP; LOSO CV)
In this approach, for each validation iteration, 
we held out variants from a single protein from the 
total set of proteins (variants from cross-validation dataset; 47 proteins),
these variants were reserved solely for model evaluation, 
while variants from the other proteins were used for model training.
#####################LOSO start################################
'''


def loso(ml_df,model_func,predict_proba=True):
    source = ml_df['source'].values 
    for iter in range(5): # number of loso cv
        ml_df_positive = ml_df[ml_df['label']!=0]
        ml_df_negative = ml_df[ml_df['label']==0]
        ml_df_negative = ml_df_negative.sample(frac=0.02) # balancing positive-negative ratio 
        ml_df_temp = pd.concat([ml_df_positive,ml_df_negative],axis=0,)
        original_index = ml_df_temp.index
        sources_all = list(Counter(list(source)).keys()) #[protein1,protein2,....]
        np.random.shuffle(sources_all)
        pred_y,test_y = [],[]
        index_order = []
        for source_name in sources_all:
            train_set = ml_df_temp[ml_df_temp['source']!=source_name]
            test_set = ml_df_temp[ml_df_temp['source']==source_name]
            train_label = train_set['label'].values
            train_matrix = train_set.drop(['label','source'],axis=1).values
            test_matrix = test_set.drop(['label','source'],axis=1).values
            scaler = MinMaxScaler()
            train_matrix = scaler.fit_transform(train_matrix)
            test_matrix = scaler.transform(test_matrix)
            test_label = test_set['label'].values
            model = model_func()
            model.fit(train_matrix,train_label)
            if predict_proba:
                predictY = model.predict_proba(test_matrix)[:,1]
            else:
                predictY = model.predict(test_matrix)
            pred_y += list(predictY)
            test_y += list(test_label)
            index_order += list(test_set.index)
            del model
        print(f'-------Leave-one-source-out AUROC AUPR {iter} iteration---')
        print(' AUROC:',roc_auc_score(test_y,pred_y)) 
        print(' AUPR:',compute_aupr(test_y,pred_y))
        order = reverse_index_map(original_index,index_order)
        pred_y = np.array(pred_y)[order]
        test_y = np.array(test_y)[order]
        print(' AUROC for single amino-acid mutation :',
              roc_auc_score(
            list(test_y[is_single_pos])+list(test_y)[ml_df_positive.shape[0]:],
            list(pred_y[is_single_pos])+list(pred_y)[ml_df_positive.shape[0]:],
            ))
        

ml_df = pd.read_csv('data/dataset/ml_features_train_eval.tsv',sep='\t') ## machine learning matrix
source_data = pd.read_csv('data/dataset/data_merged.tsv',sep='\t') ## source sequences
labels = ml_df['label'].values
labels = np.where(labels!=0,1,0)
ml_df['label'] = labels 

cv_proteins = pd.DataFrame(cross_validation_proteins,columns=['Gene_Name'])
cv_df = pd.merge(ml_df,cv_proteins,on='Gene_Name')
cv_source_file = pd.merge(source_data,cv_proteins,on='Gene_Name')
cv_source_file_positive = cv_source_file[cv_source_file['label']!=0]
mu_count = [len(str(x).split(' ')) for x in cv_source_file_positive['MuPosition'].values]
is_single_pos = [i for i in range(len(mu_count)) if mu_count[i]==1] ### filter single aa mutations

ml_df = cv_df.copy()
ml_df['source'] = ml_df['Gene_Name'].values
ml_df = ml_df.drop(['Gene_Name'],axis=1)

print('--------IP models; Leave-one-source-out CV --------')
print('---IP-SVR LOSO---')
def model_function():
    model = SVR()
    return model
loso(ml_df,model_function,predict_proba=False,)
print('---IP-LR LOSO---')
def model_function():
    model = LogisticRegression(class_weight='balanced',max_iter=50)
    return model
loso(ml_df,model_function)
print('---IP-RF LOSO---')
def model_function():
    model = RandomForestClassifier(class_weight='balanced',n_estimators=100,random_state=111)
    return model
loso(ml_df,model_function,predict_proba=True,)
del ml_df,cv_df,cv_source_file,cv_source_file_positive

'''
'Background' mutations were generated following the same IDRs: Domains ratio 
as the collected 'Impact' samples
(weighted sampling)
'''

ml_df = pd.read_csv(
    'data/dataset/ml_features_train_eval_weighted_sampling.tsv',sep='\t') ## machine learning matrix
source_data = pd.read_csv(
    'data/dataset/data_merged_weighted_sampling.tsv',sep='\t') ## source sequences
labels = ml_df['label'].values
labels = np.where(labels!=0,1,0)
ml_df['label'] = labels 

cv_proteins = pd.DataFrame(cross_validation_proteins,columns=['Gene_Name'])
cv_df = pd.merge(ml_df,cv_proteins,on='Gene_Name')
cv_source_file = pd.merge(source_data,cv_proteins,on='Gene_Name')
cv_source_file_positive = cv_source_file[cv_source_file['label']!=0]
mu_count = [len(str(x).split(' ')) for x in cv_source_file_positive['MuPosition'].values]
is_single_pos = [i for i in range(len(mu_count)) if mu_count[i]==1] ### filter single aa mutations

ml_df = cv_df.copy()
ml_df['source'] = ml_df['Gene_Name'].values
ml_df = ml_df.drop(['Gene_Name'],axis=1)

print('--------IP models; Leave-one-source-out CV; weighted sampling --------')
print('---IP-SVR LOSO weighted sampling---')
def model_function():
    model = SVR()
    return model
loso(ml_df,model_function,predict_proba=False,)
print('---IP-LR LOSO weighted sampling---')
def model_function():
    model = LogisticRegression(class_weight='balanced',max_iter=50)
    return model
loso(ml_df,model_function)
print('---IP-RF LOSO weighted sampling---')
def model_function():
    model = RandomForestClassifier(class_weight='balanced',n_estimators=100,random_state=111)
    return model
loso(ml_df,model_function,predict_proba=True,)
del ml_df,cv_df,cv_source_file,cv_source_file_positive

'''
'Background' mutations were generated aligning the impact dataset aa-ratio (AA weighted sampling)
'''

ml_df = pd.read_csv(
    'data/dataset/ml_features_train_eval_aa_weighted_sampling.tsv',sep='\t') ## machine learning matrix
source_data = pd.read_csv(
    'data/dataset/data_merged_aa_weighted_background.tsv',sep='\t') ## source sequences
labels = ml_df['label'].values
labels = np.where(labels!=0,1,0)
ml_df['label'] = labels 

cv_proteins = pd.DataFrame(cross_validation_proteins,columns=['Gene_Name'])
cv_df = pd.merge(ml_df,cv_proteins,on='Gene_Name')
cv_source_file = pd.merge(source_data,cv_proteins,on='Gene_Name')
cv_source_file_positive = cv_source_file[cv_source_file['label']!=0]
mu_count = [len(str(x).split(' ')) for x in cv_source_file_positive['MuPosition'].values]
is_single_pos = [i for i in range(len(mu_count)) if mu_count[i]==1] ### filter single aa mutations

ml_df = cv_df.copy()
ml_df['source'] = ml_df['Gene_Name'].values
ml_df = ml_df.drop(['Gene_Name'],axis=1)

print('--------IP models; Leave-one-source-out CV; AA weighted sampling --------')
print('---IP-SVR LOSO AA weighted sampling---')
def model_function():
    model = SVR()
    return model
loso(ml_df,model_function,predict_proba=False,)
print('---IP-LR LOSO AA weighted sampling---')
def model_function():
    model = LogisticRegression(class_weight='balanced',max_iter=50)
    return model
loso(ml_df,model_function)
print('---IP-RF LOSO AA weighted sampling---')
def model_function():
    model = RandomForestClassifier(class_weight='balanced',n_estimators=100,random_state=111,max_depth=5)
    return model
loso(ml_df,model_function,predict_proba=True,)
del ml_df,cv_df,cv_source_file,cv_source_file_positive


'''
##################### IP LOSO end################################
'''

'''
PSMutPred-IP validation on independent test set 
(variants corresponding to non-human proteins)
#####################validation start################################
'''
ml_df = pd.read_csv('data/dataset/ml_features_train_eval.tsv',sep='\t')
source_data = pd.read_csv('data/dataset/data_merged.tsv',sep='\t')

labels = ml_df['label'].values
labels = np.where(labels!=0,1,0)
ml_df['label'] = labels 
cv_proteins = pd.DataFrame(cross_validation_proteins,columns=['Gene_Name'])
test_proteins = pd.DataFrame(independent_test_set_proteins,columns=['Gene_Name'])
cv_df = pd.merge(ml_df,cv_proteins,on='Gene_Name')
test_df = pd.merge(ml_df,test_proteins,on='Gene_Name')
test_source_file = pd.merge(source_data,test_proteins,on='Gene_Name')
mu_count = [len(str(x).split(' ')) for x in test_source_file['MuPosition'].values]
is_single = [i for i in range(len(mu_count)) if mu_count[i]==1]
train_label = cv_df['label'].values 
train_matrix = cv_df.drop(['label','Gene_Name'],axis=1).values
test_label = test_df['label'].values 
test_matrix = test_df.drop(['label','Gene_Name'],axis=1).values

scaler = MinMaxScaler()
train_matrix = scaler.fit_transform(train_matrix)
test_matrix = scaler.transform(test_matrix)


model = SVR()
model.fit(train_matrix,train_label)
predictY = model.predict(test_matrix)
print('--------IP models; validation on independent test set --------')
print('---IP-SVR test set---')
print('AUROC:',roc_auc_score(test_label,predictY))
print('AUPR:',compute_aupr(test_label,predictY))
print('AUROC for single amino-acid mutation :',roc_auc_score(test_label[is_single],predictY[is_single]))
discriminative_power(test_label,predictY)

model = RandomForestClassifier(class_weight='balanced',n_estimators=100,max_depth=10,random_state=111)
model.fit(train_matrix,train_label)
predictY = model.predict_proba(test_matrix)[:,1]

print('---IP-RF test set---')
print('AUROC:',roc_auc_score(test_label,predictY))
print('AUPR:',compute_aupr(test_label,predictY))
print('AUROC for single amino-acid mutation :',roc_auc_score(test_label[is_single],predictY[is_single]))
discriminative_power(test_label,predictY)

# model = LogisticRegression(class_weight='balanced',max_iter=150,solver='liblinear',penalty='l1')
# model.fit(train_matrix,train_label)
# predictY = model.predict_proba(test_matrix)[:,1]
# print('---IP-LR test set---')
# print('AUROC:',roc_auc_score(test_label,predictY))
# print('AUPR:',compute_aupr(test_label,predictY))
# print('AUROC for single amino-acid mutation :',roc_auc_score(test_label[is_single],predictY[is_single]))
# discriminative_power(test_label,predictY)
# del ml_df,cv_df,test_df
# exit()
'''
'Background' mutations were generated following the same IDRs: Domains ratio 
as the collected 'Impact' samples
(weighted sampling)
'''
ml_df = pd.read_csv('data/dataset/ml_features_train_eval_weighted_sampling.tsv',sep='\t')
source_data = pd.read_csv('data/dataset/data_merged.tsv',sep='\t')

labels = ml_df['label'].values
labels = np.where(labels!=0,1,0)
ml_df['label'] = labels 
cv_proteins = pd.DataFrame(cross_validation_proteins,columns=['Gene_Name'])
test_proteins = pd.DataFrame(independent_test_set_proteins,columns=['Gene_Name'])
cv_df = pd.merge(ml_df,cv_proteins,on='Gene_Name')
test_df = pd.merge(ml_df,test_proteins,on='Gene_Name')
test_source_file = pd.merge(source_data,test_proteins,on='Gene_Name')
mu_count = [len(str(x).split(' ')) for x in test_source_file['MuPosition'].values]
is_single = [i for i in range(len(mu_count)) if mu_count[i]==1]
train_label = cv_df['label'].values 
train_matrix = cv_df.drop(['label','Gene_Name'],axis=1).values
test_label = test_df['label'].values 
test_matrix = test_df.drop(['label','Gene_Name'],axis=1).values

scaler = MinMaxScaler()
train_matrix = scaler.fit_transform(train_matrix)
test_matrix = scaler.transform(test_matrix)


model = SVR()
model.fit(train_matrix,train_label)
predictY = model.predict(test_matrix)
print('--------IP models; validation on independent test set; weighted sampling --------')
print('---IP-SVR test set weighted sampling---')
print('AUROC:',roc_auc_score(test_label,predictY))
print('AUPR:',compute_aupr(test_label,predictY))
print('AUROC for single amino-acid mutation :',roc_auc_score(test_label[is_single],predictY[is_single]))
discriminative_power(test_label,predictY)

model = RandomForestClassifier(class_weight='balanced',n_estimators=100,max_depth=10,random_state=111)
model.fit(train_matrix,train_label)
predictY = model.predict_proba(test_matrix)[:,1]

print('---IP-RF test set weighted sampling---')
print('AUROC:',roc_auc_score(test_label,predictY))
print('AUPR:',compute_aupr(test_label,predictY))
print('AUROC for single amino-acid mutation :',roc_auc_score(test_label[is_single],predictY[is_single]))
discriminative_power(test_label,predictY)

# model = LogisticRegression(class_weight='balanced',max_iter=150,solver='liblinear',penalty='l1')
# model.fit(train_matrix,train_label)
# predictY = model.predict_proba(test_matrix)[:,1]
# print('---IP-LR test set weighted sampling---')
# print('AUROC:',roc_auc_score(test_label,predictY))
# print('AUPR:',compute_aupr(test_label,predictY))
# print('AUROC for single amino-acid mutation :',roc_auc_score(test_label[is_single],predictY[is_single]))
# discriminative_power(test_label,predictY)
del ml_df,cv_df,test_df

'''
'Background' mutations were generated following the same AA substitution rate
as the collected 'Impact' samples
(weighted sampling)
'''
ml_df = pd.read_csv('data/dataset/ml_features_train_eval_aa_weighted_sampling.tsv',sep='\t')
source_data = pd.read_csv('data/dataset/data_merged_aa_weighted_background.tsv',sep='\t')

labels = ml_df['label'].values
labels = np.where(labels!=0,1,0)
ml_df['label'] = labels 
cv_proteins = pd.DataFrame(cross_validation_proteins,columns=['Gene_Name'])
test_proteins = pd.DataFrame(independent_test_set_proteins,columns=['Gene_Name'])
cv_df = pd.merge(ml_df,cv_proteins,on='Gene_Name')
test_df = pd.merge(ml_df,test_proteins,on='Gene_Name')
test_source_file = pd.merge(source_data,test_proteins,on='Gene_Name')
mu_count = [len(str(x).split(' ')) for x in test_source_file['MuPosition'].values]
is_single = [i for i in range(len(mu_count)) if mu_count[i]==1]
train_label = cv_df['label'].values 
train_matrix = cv_df.drop(['label','Gene_Name'],axis=1).values
test_label = test_df['label'].values 
test_matrix = test_df.drop(['label','Gene_Name'],axis=1).values

scaler = MinMaxScaler()
train_matrix = scaler.fit_transform(train_matrix)
test_matrix = scaler.transform(test_matrix)


model = SVR()
model.fit(train_matrix,train_label)
predictY = model.predict(test_matrix)
print('--------IP models; validation on independent test set;AA weighted sampling --------')
print('---IP-SVR test set AA weighted sampling---')
print('AUROC:',roc_auc_score(test_label,predictY))
print('AUPR:',compute_aupr(test_label,predictY))
print('AUROC for single amino-acid mutation :',roc_auc_score(test_label[is_single],predictY[is_single]))
discriminative_power(test_label,predictY)

model = RandomForestClassifier(class_weight='balanced',n_estimators=200,random_state=111)
model.fit(train_matrix,train_label)
predictY = model.predict_proba(test_matrix)[:,1]

print('---IP-RF test set AA weighted sampling---')
print('AUROC:',roc_auc_score(test_label,predictY))
print('AUPR:',compute_aupr(test_label,predictY))
print('AUROC for single amino-acid mutation :',roc_auc_score(test_label[is_single],predictY[is_single]))
discriminative_power(test_label,predictY)

# model = LogisticRegression(class_weight='balanced',max_iter=150,solver='liblinear',penalty='l1')
# model.fit(train_matrix,train_label)
# predictY = model.predict_proba(test_matrix)[:,1] 
# print('---IP-LR test set AA weighted sampling---')
# print('AUROC:',roc_auc_score(test_label,predictY))
# print('AUPR:',compute_aupr(test_label,predictY))
# print('AUROC for single amino-acid mutation :',roc_auc_score(test_label[is_single],predictY[is_single]))
# discriminative_power(test_label,predictY)
del ml_df,cv_df,test_df

'''
##################### IP validation end################################
'''



'''
PSMutPred-SP leave-one-source-out cross-validation (SP; LOSO CV)
In this approach, for each validation iteration, 
we held out variants from a single protein from the 
total set of proteins (variants from cross-validation dataset; 47 proteins),
these variants were reserved solely for model evaluation, 
while variants from the other proteins were used for model training.
'''

def loso(ml_df,source,labels,model_func,predict_proba=True):
    ml_df['source'],ml_df['label'] = source,labels
    sources_all = list(Counter(list(source)).keys())
    np.random.shuffle(sources_all)
    pred_y,test_y = [],[]
    index_order = []
    for source_name in sources_all:
        train_set = ml_df[ml_df['source']!=source_name]
        test_set = ml_df[ml_df['source']==source_name]
        train_label = train_set['label'].values     
        train_matrix = train_set.drop(['label','source'],axis=1).values
        test_matrix = test_set.drop(['label','source'],axis=1).values
        test_label = test_set['label'].values
        model = model_func()
        model.fit(train_matrix,train_label)
        if predict_proba:
            predictY = model.predict_proba(test_matrix)[:,1]
        else:
            predictY = model.predict(test_matrix)
        pred_y += list(predictY)
        test_y += list(test_label)
        index_order += list(test_set.index)
        del model
    print(' AUROC:',roc_auc_score(test_y,pred_y)) 
    print(' AUPR:',compute_aupr(test_y,pred_y))
    discriminative_power(test_y,pred_y)
    
    

    

ml_df = pd.read_csv('data/dataset/ml_features_train_eval.tsv',sep='\t')
ml_df = ml_df[ml_df['label']!=0]
source_data = pd.read_csv('data/dataset/data_merged.tsv',sep='\t')
source_data = source_data[source_data['label']!=0]
labels = ml_df['label'].values
labels = np.where(labels==2,1,0) # 2-‘strengthen' 1'weaken'
ml_df['label'] = labels 

cv_proteins = pd.DataFrame(cross_validation_proteins,columns=['Gene_Name'])
cv_df = pd.merge(ml_df,cv_proteins,on='Gene_Name')
cv_source_file = pd.merge(source_data,cv_proteins,on='Gene_Name')

ml_df = cv_df.copy()
source,labels = ml_df['Gene_Name'].values,ml_df['label'].values
ml_df = ml_df.drop(['label','Gene_Name'],axis=1)
print('--------Leave one source out CV; strengthen/weaken prediction (SP, LOSO)-------')
print('---SP-LR---')
def model_function():
    model = LogisticRegression(class_weight='balanced',max_iter=150,penalty='l1',solver='liblinear',C=0.3)
    return model
loso(ml_df,source,labels,model_function,)
print('---SP-RF---')
def model_function():
    model = RandomForestClassifier(class_weight='balanced',n_estimators=100,random_state=111,max_depth=5)
    return model
loso(ml_df,source,labels,model_function,predict_proba=True)
print('---SP-SVR---')
def model_function():
    model = SVR()
    return model
loso(ml_df,source,labels,model_function,predict_proba=False,)
del ml_df,cv_df


'''
PSMutPred-SP validation on independent test set 
(variants corresponding to non-human proteins)
'''
ml_df = pd.read_csv('data/dataset/ml_features_train_eval.tsv',sep='\t')
ml_df = ml_df[ml_df['label']!=0]
labels = ml_df['label'].values
labels = np.where(labels==2,1,0)
ml_df['label'] = labels 

cv_proteins = pd.DataFrame(cross_validation_proteins,columns=['Gene_Name'])
test_proteins = pd.DataFrame(independent_test_set_proteins,columns=['Gene_Name'])
cv_df = pd.merge(ml_df,cv_proteins,on='Gene_Name')
test_df = pd.merge(ml_df,test_proteins,on='Gene_Name')

train_label = cv_df['label'].values 
train_matrix = cv_df.drop(['label','Gene_Name'],axis=1).values
test_label = test_df['label'].values 
test_matrix = test_df.drop(['label','Gene_Name'],axis=1).values

scaler = MinMaxScaler()
train_matrix = scaler.fit_transform(train_matrix)
test_matrix = scaler.transform(test_matrix)

test_results_sp = test_df.copy()


model = LogisticRegression(class_weight='balanced',max_iter=250,penalty='l1',solver='liblinear',)
model.fit(train_matrix,train_label)
predictY = model.predict_proba(test_matrix)[:,1]
test_results_sp['SP-LR'] = predictY
print('-------strengthen/weaken prediction validation------')
print('-------test set Logistic Regression-------')
print('AUROC:',roc_auc_score(test_label,predictY))

model = RandomForestClassifier(class_weight='balanced',n_estimators=100,max_depth=13,random_state=111)
model.fit(train_matrix,train_label)
predictY = model.predict_proba(test_matrix)[:,1]
print('-------test set Random Forest-------')
test_results_sp['SP-RF'] = predictY
print('AUROC:',roc_auc_score(test_label,predictY))

model = SVR()
model.fit(train_matrix,train_label)
predictY = model.predict(test_matrix)
print('-------test set SVR-------')
test_results_sp['SP-SVR'] = predictY
print('AUROC:',roc_auc_score(test_label,predictY))

