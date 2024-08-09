print('--------preparing models--------')
from modules import *
from ml_feature_construction import *
import bisect 
submodel_path = 'data/models/submodels/'
class lr_merge_model:
    def __init__(self):   
        self.lr_model1 = joblib.load(os.path.join(submodel_path,'lr_sub_1.model'))
        self.lr_model2 = joblib.load(os.path.join(submodel_path,'lr_sub_2.model'))
        self.lr_model3 = joblib.load(os.path.join(submodel_path,'lr_sub_3.model'))
        self.lr_model4 = joblib.load(os.path.join(submodel_path,'lr_sub_4.model'))
        self.lr_model5 = joblib.load(os.path.join(submodel_path,'lr_sub_5.model'))
        self.lr_model6 = joblib.load(os.path.join(submodel_path,'lr_sub_6.model'))
        self.lr_model7 = joblib.load(os.path.join(submodel_path,'lr_sub_7.model'))
        self.lr_model8 = joblib.load(os.path.join(submodel_path,'lr_sub_8.model'))
        self.lr_model9 = joblib.load(os.path.join(submodel_path,'lr_sub_9.model'))
        self.lr_model10 = joblib.load(os.path.join(submodel_path,'lr_sub_10.model'))
    

    def predict(self,input_matrix):
        p1 = self.lr_model1.predict_proba(input_matrix)[:,1]
        p2 = self.lr_model2.predict_proba(input_matrix)[:,1]
        p3 = self.lr_model3.predict_proba(input_matrix)[:,1]
        p4 = self.lr_model4.predict_proba(input_matrix)[:,1]
        p5 = self.lr_model5.predict_proba(input_matrix)[:,1]
        p6 = self.lr_model6.predict_proba(input_matrix)[:,1]
        p7 = self.lr_model7.predict_proba(input_matrix)[:,1]
        p8 = self.lr_model8.predict_proba(input_matrix)[:,1]
        p9 = self.lr_model9.predict_proba(input_matrix)[:,1]
        p10 = self.lr_model10.predict_proba(input_matrix)[:,1]
        return (p1+p2+p3+p4+p5+p6+p7+p8+p9+p10)/10


    
class rf_merge_model:
    def __init__(self):   
        self.rf_model1 = joblib.load(os.path.join(submodel_path,'rf_sub_1.model'))
        self.rf_model2 = joblib.load(os.path.join(submodel_path,'rf_sub_2.model'))
        self.rf_model3 = joblib.load(os.path.join(submodel_path,'rf_sub_3.model'))
        self.rf_model4 = joblib.load(os.path.join(submodel_path,'rf_sub_4.model'))
        self.rf_model5 = joblib.load(os.path.join(submodel_path,'rf_sub_5.model'))
        self.rf_model6 = joblib.load(os.path.join(submodel_path,'rf_sub_6.model'))
        self.rf_model7 = joblib.load(os.path.join(submodel_path,'rf_sub_7.model'))
        self.rf_model8 = joblib.load(os.path.join(submodel_path,'rf_sub_8.model'))
        self.rf_model9 = joblib.load(os.path.join(submodel_path,'rf_sub_9.model'))
        self.rf_model10 = joblib.load(os.path.join(submodel_path,'rf_sub_10.model'))
    

    def predict(self,input_matrix):
        p1 = self.rf_model1.predict_proba(input_matrix)[:,1]
        p2 = self.rf_model2.predict_proba(input_matrix)[:,1]
        p3 = self.rf_model3.predict_proba(input_matrix)[:,1]
        p4 = self.rf_model4.predict_proba(input_matrix)[:,1]
        p5 = self.rf_model5.predict_proba(input_matrix)[:,1]
        p6 = self.rf_model6.predict_proba(input_matrix)[:,1]
        p7 = self.rf_model7.predict_proba(input_matrix)[:,1]
        p8 = self.rf_model8.predict_proba(input_matrix)[:,1]
        p9 = self.rf_model9.predict_proba(input_matrix)[:,1]
        p10 = self.rf_model10.predict_proba(input_matrix)[:,1]
        return (p1+p2+p3+p4+p5+p6+p7+p8+p9+p10)/10
    

class svr_merge_model:
    def __init__(self):   
        self.svr_model1 = joblib.load(os.path.join(submodel_path,'svr_sub_1.model'))
        self.svr_model2 = joblib.load(os.path.join(submodel_path,'svr_sub_2.model'))
        self.svr_model3 = joblib.load(os.path.join(submodel_path,'svr_sub_3.model'))
        self.svr_model4 = joblib.load(os.path.join(submodel_path,'svr_sub_4.model'))
        self.svr_model5 = joblib.load(os.path.join(submodel_path,'svr_sub_5.model'))
        self.svr_model6 = joblib.load(os.path.join(submodel_path,'svr_sub_6.model'))
        self.svr_model7 = joblib.load(os.path.join(submodel_path,'svr_sub_7.model'))
        self.svr_model8 = joblib.load(os.path.join(submodel_path,'svr_sub_8.model'))
        self.svr_model9 = joblib.load(os.path.join(submodel_path,'svr_sub_9.model'))
        self.svr_model10 = joblib.load(os.path.join(submodel_path,'svr_sub_10.model'))
    

    def predict(self,input_matrix):
        p1 = self.svr_model1.predict(input_matrix)
        p2 = self.svr_model2.predict(input_matrix)
        p3 = self.svr_model3.predict(input_matrix)
        p4 = self.svr_model4.predict(input_matrix)
        p5 = self.svr_model5.predict(input_matrix)
        p6 = self.svr_model6.predict(input_matrix)
        p7 = self.svr_model7.predict(input_matrix)
        p8 = self.svr_model8.predict(input_matrix)
        p9 = self.svr_model9.predict(input_matrix)
        p10 = self.svr_model10.predict(input_matrix)
        return (p1+p2+p3+p4+p5+p6+p7+p8+p9+p10)/10


def point_mutation(seq,point,before,after):
    if seq[point-1] != before:
        return False
    else:
        return seq[:point-1]+after+seq[point:]

def convert2Predictable(dat,wt_seq,gene_name):
    mt_aas = dat['mt_aa'].values
    wt_aas = dat['wt_aa'].values
    positions = dat['position'].values
    mt_seqs = []
    size = len(mt_aas)
    for pos,wt_aa,mt_aa in zip(positions,wt_aas,mt_aas):
        mt_seq = point_mutation(wt_seq, int(pos), wt_aa, mt_aa)
        mt_seqs.append(mt_seq)
        
    dat['MT_AA'] = mt_aas # 根据新的buildfeature文件重命名
    dat['WT_AA'] = wt_aas 
    dat['Seq'] = mt_seqs
    dat['wt_seq'] = [wt_seq]*size
    dat['MuPosition'] = dat['position'].apply(lambda x:str(x))
    dat['gene_name'] = [gene_name]*size
    return dat

def predict(dat,wt_seq,gene_name):
    dat1 = convert2Predictable(dat,wt_seq,gene_name)
    dat1 = build_feature_df(dat1)
    dat_numeric = dat1._get_numeric_data()
    dat2predict_model1 = dat_numeric
    dat2predict_model1 = scaler.transform(dat2predict_model1)
    dat2predict_model2 = dat_numeric
    dat2predict_model2 = scaler_lr.transform(dat2predict_model2)
    dat['pred_ip_lr'] = lr_m1.predict(dat2predict_model1)
    dat['pred_ip_rf'] = rf_m1.predict(dat_numeric)
    dat['pred_ip_svr'] = svr_m1.predict(dat2predict_model1)
    dat['pred_sp_lr'] = lr_m2.predict_proba(dat2predict_model2)[:,1]
    dat['pred_sp_rf'] = rf_m2.predict_proba(dat_numeric)[:,1]
    dat.drop(['Seq','wt_seq','MuPosition','MT_AA','WT_AA'],axis=1,inplace=True,)
    return dat

def predict_df(input_df,uniprot_entry_name):
    if uniprot_entry_name not in uniprotID2geneName.keys():
        raise Exception(f'{uniprot_entry_name} not found or type error!! Input 2 has to be UNIPROT ENTRY NAME, eg.EPS8_HUMAN')
    gene_name__ = uniprotID2geneName[uniprot_entry_name]
    seq__ = uniprotID2seq[uniprot_entry_name]
    for wt_aa_,pos_,mt_aa_ in zip(
        input_df['wt_aa'].values,input_df['position'].values,input_df['mt_aa'].values):
        if seq__[pos_-1] != wt_aa_:
            raise Exception('variants position error')
        if mt_aa_ not in RESIDUES:
            raise Exception('Wrong amino acid character')
    df_ = predict(input_df,seq__,gene_name__)
    df_['rank_ip_lr'] = df_['pred_ip_lr'].map(lambda x:sort_lr(x))
    df_['rank_ip_rf'] = df_['pred_ip_rf'].map(lambda x:sort_rf(x))
    df_['rank_ip_svr'] = df_['pred_ip_svr'].map(lambda x:sort_svr(x))
    df_['rank_sp_lr'] = df_['pred_ip_lr'].map(lambda x:sort_lr2(x))
    df_['rank_sp_rf'] = df_['pred_ip_rf'].map(lambda x:sort_rf2(x))
    return df_

sequence_all_file = 'data/dataset/uniprot_human.fasta'
uniprotID2geneName,uniprotID2seq = {},{}
for record in SeqIO.parse(sequence_all_file,'fasta'):
    uniprotID = record.id.split('|')[-1]
    geneName = record.description.split('GN=')[-1].split(' ')[0]
    uniprotID2geneName[uniprotID] = geneName 
    uniprotID2seq[uniprotID] = str(record.seq)

lr_m1 = lr_merge_model()
rf_m1 = rf_merge_model()
svr_m1 = svr_merge_model()
lr_m2 = joblib.load('data/models/sp_lr.model')
rf_m2 = joblib.load('data/models/sp_rf.model')
scaler = joblib.load('data/models/ip.scaler')
scaler_lr = joblib.load('data/models/sp.scaler')


random_psmutpred_scores_dict = joblib.load('data/utils/random_psmutpred_scores_reduced.dict')
full_lr_predicted = random_psmutpred_scores_dict['IP-LR']
full_rf_predicted = random_psmutpred_scores_dict['IP-RF']
full_svr_predicted = random_psmutpred_scores_dict['IP-SVR']
full_lr2_predicted = random_psmutpred_scores_dict['SP-LR']
full_rf2_predicted = random_psmutpred_scores_dict['SP-RF']

def sort_lr(value):
    rank_avg = 0
    len_sorted = len(full_lr_predicted)
    rank_avg += bisect.bisect_left(full_lr_predicted,value) 
    return rank_avg/len_sorted

def sort_rf(value):
    rank_avg = 0
    len_sorted = len(full_rf_predicted)
    rank_avg += bisect.bisect_left(full_rf_predicted,value) 
    return rank_avg/len_sorted

def sort_svr(value):
    rank_avg = 0
    len_sorted = len(full_svr_predicted)
    rank_avg += bisect.bisect_left(full_svr_predicted,value) 
    return rank_avg/len_sorted

def sort_lr2(value):
    rank_avg = 0
    len_sorted = len(full_lr2_predicted)
    rank_avg += bisect.bisect_left(full_lr2_predicted,value) 
    return rank_avg/len_sorted

def sort_rf2(value):
    rank_avg = 0
    len_sorted = len(full_rf2_predicted)
    rank_avg += bisect.bisect_left(full_rf2_predicted,value) 
    return rank_avg/len_sorted


######### updated new models by using random background dataset following the same aa mutant ratio as impact dataset ###############
class rf_merge_model_aa_weighted:
    def __init__(self):
        scaler_lst = []
        model_lst = []
        for idx in range(1,11):
            idx = str(idx)
            scaler_lst.append(joblib.load(f'data/models/submodels/scaler_aa_weighted_sample_sub_{idx}.model'))
            model_lst.append(joblib.load(f'data/models/submodels/rf_aa_weighted_sample_sub_{idx}.model'))
        self.scalers = scaler_lst
        self.models = model_lst

    def predict_(self,input_matrix):
        predicted = []
        for scaler_,model_ in zip(self.scalers,self.models):
            input_matrix_trans = input_matrix.copy()
            input_matrix_trans = scaler_.transform(input_matrix_trans)
            scores = model_.predict_proba(input_matrix_trans)[:,1]
            predicted.append(scores)
            
        return np.average(np.array(predicted),axis=0)

   
class svr_merge_model_aa_weighted:
    def __init__(self):
        scaler_lst = []
        model_lst = []
        for idx in range(1,11):
            idx = str(idx)
            scaler_lst.append(joblib.load(f'data/models/submodels/scaler_aa_weighted_sample_sub_{idx}.model'))
            model_lst.append(joblib.load(f'data/models/submodels/svr_aa_weighted_sample_sub_{idx}.model'))
        
        self.scalers = scaler_lst
        self.models = model_lst

    def predict_(self,input_matrix):
        predicted = []
        for scaler_,model_ in zip(self.scalers,self.models):
            input_matrix_trans = input_matrix.copy()
            input_matrix_trans = scaler_.transform(input_matrix_trans)
            scores = model_.predict(input_matrix_trans)
            predicted.append(scores)
        return np.average(np.array(predicted),axis=0)
    
rf_aa_weighted = rf_merge_model_aa_weighted()
svr_aa_weighted = svr_merge_model_aa_weighted()
clinvar_missense_all_add_rf_svr_aa_weighted = pd.read_csv('data/utils/clinvar_missense_all_add_rf_svr_aa_weighted.tsv',sep='\t')
## data for generating rank score
full_rf_aa_weighted_predicted = list(clinvar_missense_all_add_rf_svr_aa_weighted['ip_rf_aa_weighted'].values) 
full_svr_aa_weighted_predicted = list(clinvar_missense_all_add_rf_svr_aa_weighted['ip_svr_aa_weighted'].values)
from scipy.stats import percentileofscore
def sort_rf_aw(value):
    percentile = percentileofscore(full_rf_aa_weighted_predicted,
                                   value, kind='weak')
    return percentile/100

def sort_svr_aw(value):
    percentile = percentileofscore(full_svr_aa_weighted_predicted,
                                   value, kind='weak')
    return percentile/100   


def predict(dat,wt_seq,gene_name):
    dat1 = convert2Predictable(dat,wt_seq,gene_name)
    dat1 = build_feature_df(dat1)
    dat_numeric = dat1._get_numeric_data()
    dat2predict_model1 = dat_numeric
    dat2predict_model1 = scaler.transform(dat2predict_model1)
    dat2predict_model2 = dat_numeric
    dat2predict_model2 = scaler_lr.transform(dat2predict_model2)
    
    dat['pred_aw_ip_rf'] = rf_aa_weighted.predict_(dat_numeric)# in-build normalization
    dat['pred_aw_ip_svr'] = svr_aa_weighted.predict_(dat_numeric)# in-build normalization
    dat['pred_aw_ip_rf_rank'] = dat['pred_aw_ip_rf'].apply(lambda x:sort_rf_aw(x))
    dat['pred_aw_ip_svr_rank'] = dat['pred_aw_ip_svr'].apply(lambda x:sort_svr_aw(x))
    
    dat['pred_ip_lr'] = lr_m1.predict(dat2predict_model1)
    dat['pred_ip_rf'] = rf_m1.predict(dat_numeric)
    dat['pred_ip_svr'] = svr_m1.predict(dat2predict_model1)
    dat['pred_sp_lr'] = lr_m2.predict_proba(dat2predict_model2)[:,1]
    dat['pred_sp_rf'] = rf_m2.predict_proba(dat_numeric)[:,1]
    dat.drop(['Seq','wt_seq','MuPosition','MT_AA','WT_AA'],axis=1,inplace=True,)
    return dat

def predict_df(input_df,uniprot_entry_name):
    if uniprot_entry_name not in uniprotID2geneName.keys():
        raise Exception(f'{uniprot_entry_name} not found or type error!! Input 2 has to be UNIPROT ENTRY NAME, eg.EPS8_HUMAN')
    gene_name__ = uniprotID2geneName[uniprot_entry_name]
    seq__ = uniprotID2seq[uniprot_entry_name]
    for wt_aa_,pos_,mt_aa_ in zip(
        input_df['wt_aa'].values,input_df['position'].values,input_df['mt_aa'].values):
        if seq__[pos_-1] != wt_aa_:
            raise Exception('variants position error')
        if mt_aa_ not in RESIDUES:
            raise Exception('Wrong amino acid character')
    df_ = predict(input_df,seq__,gene_name__)
    df_['rank_ip_lr'] = df_['pred_ip_lr'].map(lambda x:sort_lr(x))
    df_['rank_ip_rf'] = df_['pred_ip_rf'].map(lambda x:sort_rf(x))
    df_['rank_ip_svr'] = df_['pred_ip_svr'].map(lambda x:sort_svr(x))
    df_['rank_sp_lr'] = df_['pred_ip_lr'].map(lambda x:sort_lr2(x))
    df_['rank_sp_rf'] = df_['pred_ip_rf'].map(lambda x:sort_rf2(x))
    
    return df_
print('-----preparing models completed------')
