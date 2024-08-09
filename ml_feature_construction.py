from modules import *
from data.utils.iupred3.iupred3_lib import iupred
from data.utils.Pscore.predict_residue_score_only import position_scores

'''
data preparation
'''
RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
pfam_all = pd.read_csv('data/utils/pfam_human_merged.tsv',sep='\t') # mapping domain / IDR using pfamscan
pfam_all.drop(['pfamID','Uniprot_ID'],axis=1,inplace=True) 
pfam_dict = joblib.load('data/utils/train_eval_pfam.dict')
aa_proper_df = pd.read_csv('data/utils/aa_properties_stdNorm.txt',sep=',') # amino acid properties
aas = aa_proper_df['AA'].values
H_dict,VSC_dict,P1_dict,P2_dict,SASA_dict,NCISC_dict = {},{},{},{},{},{}
for x,aa in enumerate(aas):
    H_dict[aa] = aa_proper_df['H'].values[x]
    VSC_dict[aa] = aa_proper_df['VSC'].values[x]
    P1_dict[aa] = aa_proper_df['P1'].values[x]
    P2_dict[aa] = aa_proper_df['P2'].values[x]
    SASA_dict[aa] = aa_proper_df['SASA'].values[x]
    NCISC_dict[aa] = aa_proper_df['NCISC'].values[x]
props = {'H':H_dict,'VSC':VSC_dict,'P1':P1_dict,'SASA':SASA_dict,'NCISC':NCISC_dict}

neg_charge_aa = ['D','E']
pos_charge_aa = ['R','H','K']
sp2_aa = ['W','F','Y','H','R','Q','N','E','D']
aromatic_aa = ['F','W','Y','H']

'''
functions
'''
def get_prop_average(seq,property_dict):
    return sum([property_dict[x] for x in list(seq)])/len(seq)

def points_mutation(seq,positions,before,after):
    '''build mutant sequence'''
    for idx,pos in enumerate(positions):
        if seq[pos-1] != before[idx]:
            return False
        else:
            seq = seq[:pos-1] + after[idx] + seq[pos:]
    return seq


def map_pfam(gene_name,seq,pfam_f):
    '''
    output:
    binary lst (lengthen == seq)
    '''
    in_domain_lst = [0]*len(seq)
    if pfam_f.shape[0] == 0:
        return in_domain_lst
    select_rows = pfam_f[pfam_f['seq id']==gene_name]
    if select_rows.shape[0] == 0:
        return in_domain_lst
    for _,row in select_rows.iterrows():
        start,end = row['alignment start'],row['alignment end']
        if start >= end or end>=len(seq):
            continue
        in_domain_lst[start-1:end] = [1]*(end-start+1)
    return in_domain_lst

def pos_pfam_distance(gene_name,position,pfam_f):
    '''
    compute distance to IDR
    'out' -> 'IDR'; 'domain' -> 'Domain'
    '''
    if pfam_f.shape[0] == 0:
        return 'out',0
    select_rows = pfam_f[pfam_f['seq id']==gene_name]
    if select_rows.shape[0] == 0:
        return 'out',0
    min_distance = 9999
    position = int(position)
    for idx in range(select_rows.shape[0]):
        line = select_rows.iloc[idx,:]
        line = list(line)
        if min(abs(position - line[1]),abs(position-line[2]))<min_distance:
            min_distance = min(abs(position - line[1]),abs(position-line[2]))
    for idx in range(select_rows.shape[0]):
        line = select_rows.iloc[idx,:]
        line = list(line)
        if position >= line[1] and position <= line[2]:
            return 'domain',min_distance
    return 'out',0

class buildFeatures:
    def __init__(self):
        self.return_dict = {}
        self.excute = [
            'self.analyze(input_dict)',
            'self.aa_property_feat()',
            'self.pipi_feat()',
            'self.idr_feat()',
            'self.build_mut_aa_wt_features()',
            'self.build_idr_features()',
        ]

    def call(self,input_dict):
        for e in self.excute:
            exec(e)

    def analyze(self, input_dict):
        self.input_dict = input_dict
        self.seq = str(input_dict['Seq'])
        self.seq_wt = str(input_dict['wt_seq'])
        self.seq_len = len(self.seq)
        MuP = str(input_dict['MuPosition'])
        self.MuP = list(map(int,MuP.split(' ')))
        self.MuP_wt = self.MuP
        self.MuPAA = [self.seq[x-1] for x in self.MuP]
        self.MuPAA_wt = [self.seq_wt[x-1] for x in self.MuP_wt]
        self.pscore_dict = input_dict['pi_dict']
        self.pfam_list = input_dict['pfam_info']
        assert len(self.pfam_list) == len(self.seq_wt)
        self.in_pfam = input_dict['in_pfam']
        self.idr_dist = input_dict['dist2IDR']/self.seq_len
        self.idr_wt_seq = [self.seq_wt[idx] for idx in range(len(self.seq_wt)) if self.pfam_list[idx]==0]

    def aa_property_feat(self):
        self.return_dict['wt_negcharge'] = 0
        self.return_dict['wt_poscharge'] = 0
        self.return_dict['wt_sp2aa'] = 0
        self.return_dict['mt_negcharge'] = 0
        self.return_dict['mt_poscharge'] = 0
        self.return_dict['mt_sp2aa'] = 0
        for aa in self.MuPAA_wt:
            if aa in neg_charge_aa:
                self.return_dict['wt_negcharge'] = 1
            if aa in pos_charge_aa:
                self.return_dict['wt_poscharge'] = 1
            if aa in sp2_aa:
                self.return_dict['wt_sp2aa'] = 1
        for aa in self.MuPAA:
            if aa in neg_charge_aa:
                self.return_dict['mt_negcharge'] = 1
            if aa in pos_charge_aa:
                self.return_dict['mt_poscharge'] = 1
            if aa in sp2_aa:
                self.return_dict['mt_sp2aa'] = 1

    def pipi_feat(self):
        pipi_contact_lists = []
        for p in self.MuP_wt:
            p = min(max(self.pscore_dict.keys()),p)
            p = max(2,p)
            pipi_contact = self.pscore_dict[p-1][1:9]
            pipi_contact_lists.append(pipi_contact)
        if len(pipi_contact_lists) > 1:
            pipi_contact = list(np.average(np.array(pipi_contact_lists),axis=0))
        for idx,name in enumerate(['AVG_BB_SRZ','SRBB_FbyG','AVG_BB_LRZ','LRBB_FbyG',\
            'AVG_SC_SRZ','SRSC_FbyG','AVG_SC_LRZ','LRSC_FbyG']):
            self.return_dict[name] = pipi_contact[idx]

    def idr_feat(self):
        if self.in_pfam == 'out':
            self.return_dict['in_idr'] = 1
            self.return_dict['dist2idr'] = 0
        else:
            self.return_dict['in_idr'] = 0
            self.return_dict['dist2idr'] = self.idr_dist
        if self.pfam_list.count(0) == 0:
            self.return_dict['has_idr'] = 0
        else:
            self.return_dict['has_idr'] = 1

    def build_mut_aa_wt_features(self):
        for f in props.keys():
            f_list = props[f]
            mu_aa_feature_avg = get_prop_average(self.MuPAA, f_list)
            wt_aa_feature_avg = get_prop_average(self.MuPAA_wt, f_list)
            self.return_dict[f+'_wt_AA'] = wt_aa_feature_avg
            self.return_dict[f+'_diff_with_self'] = (mu_aa_feature_avg - wt_aa_feature_avg)*len(self.MuPAA)
            if len(self.idr_wt_seq) == 0: 
                self.return_dict[f+'_wt_idr'] = 0
                self.return_dict[f+'_diff_with_idr'] = 0
            else:
                idr_seq_aa_feature_avg = get_prop_average(self.idr_wt_seq, f_list)
                self.return_dict[f+'_wt_idr'] = idr_seq_aa_feature_avg
                self.return_dict[f+'_diff_with_idr'] = (mu_aa_feature_avg - idr_seq_aa_feature_avg)*len(self.MuPAA)

    def build_idr_features(self):
        iupred_wt = self.input_dict['wt_idrpred']
        self.return_dict['iupred_wt_avg'] = sum(iupred_wt)/len(self.seq_wt)
        self.return_dict['idr_wt_AA'] = np.average([iupred_wt[pos-1] for pos in self.MuP_wt])


def build_feature_df(input_dataframe,seq=None,
                     iupred_scores=None):
    #### process data to predict
    if iupred_scores and seq:
        iupred_results = {seq:iupred_scores}
    else:
        iupred_results = {}
    pscore_results,pfam_list_list = {},{}
    ml_df = pd.DataFrame()
    print('---building matrix---')
    for _,row in tqdm(input_dataframe.iterrows()):
        mt_aas,wt_aas = row['MT_AA'],row['WT_AA']
        position = row['MuPosition']
        try:
            mt_seq = row['Seq']
        except:
            mt_seq = points_mutation(row['wt_seq'],position,wt_aas,mt_aas)
            row['Seq'] = mt_seq

        try:
            wt_seq = row['wt_seq']
        except:
            wt_seq = points_mutation(row['Seq'],position,mt_aas,wt_aas)
            row['wt_seq'] = wt_seq

        if wt_seq not in iupred_results.keys():
            wt_idrpred = iupred(wt_seq,'glob')[0]
            iupred_results[wt_seq] = wt_idrpred
        else:
            wt_idrpred = iupred_results[wt_seq]
        row['wt_idrpred'] = wt_idrpred

        if wt_seq not in pscore_results.keys():
            wt_pipi_dict = position_scores(wt_seq)
            pscore_results[wt_seq] = wt_pipi_dict
        else:
            wt_pipi_dict = pscore_results[wt_seq]
        row['pi_dict'] = wt_pipi_dict
        if wt_seq not in pfam_list_list.keys():
            gene_name = row['gene_name']
            pfam_lst = map_pfam(gene_name, wt_seq, pfam_all)
            pfam_list_list[wt_seq] = pfam_lst
        else:
            pfam_lst = pfam_list_list[wt_seq]
        row['pfam_info'] = pfam_lst # [0,0,0...,1,1,1,,...0,0,0]
        in_pfam, dist = pos_pfam_distance(gene_name,position,pfam_all[pfam_all['seq id']==row['gene_name']])
        row['in_pfam'] = in_pfam
        row['dist2IDR'] = dist
        bf = buildFeatures()
        bf.call(row)
        return_dict = bf.return_dict
        ml_df = ml_df.append(return_dict,ignore_index=True)
    return ml_df

if __name__=='__main__':
    ### process training and evaluation data
    file_names = ['data_merged.tsv','data_merged_weighted_sampling.tsv']
    files_to_save = [
        'ml_features_train_eval.tsv','ml_features_train_eval_weighted_sampling.tsv'
        ] # data for evaluation
    for input_df_name,output_df_name in zip(file_names,files_to_save):
        pd_ = pd.read_csv(
            f'data/dataset/{input_df_name}',
            sep='\t')
        ml_df = pd.DataFrame()
        seq_pfam_info,seq_pfam_mapped,seq_pfam_dist = {},{},{} ### acceleration using dict
        iupred_results,pscore_results = {},{} ### acceleration using dict
        for idx,row in tqdm(pd_.iterrows()):
            if row['wt_seq'] in pfam_dict.keys():
                pfam_df = pfam_dict[row['wt_seq']]
                seq_id = pfam_df['seq id'].values[0]
            elif row['Seq'] in pfam_dict.keys():
                pfam_df = pfam_dict[row['Seq']]
                seq_id = pfam_df['seq id'].values[0]
            else:
                seq_id,pfam_df = '',pd.DataFrame() ### no pfam domain
            seq = row['Seq']
            seq_pfam_info[seq] = map_pfam(seq_id, seq, pfam_df)
            mapped,min_dist = 'out',0
            for pos in str(row['MuPosition']).split(' '): ### computing distance to domain boundary
                if pos_pfam_distance(seq_id, int(pos),pfam_df)[0] != 'out':
                    mapped = 'domain'
                min_dist += pos_pfam_distance(seq_id, int(pos),pfam_df)[1]
            if mapped!= 'domain':
                seq_pfam_dist[seq],seq_pfam_mapped[seq] = 0,'out'
            else:
                seq_pfam_dist[seq],seq_pfam_mapped[seq] = min_dist/len(str(row['MuPosition']).split(' ')),mapped

            mt_aas,wt_aas = row['MT_AA'].split(' '),row['WT_AA'].split(' ')
            positions = [int(x) for x in row['MuPosition'].split(' ')]
            mt_seq = row['Seq']
            wt_seq = points_mutation(mt_seq,positions,mt_aas,wt_aas) # possitive sample only have mt_seq
            row['wt_seq'] = wt_seq
            if wt_seq not in iupred_results.keys():
                wt_idrpred = iupred(wt_seq,'glob')[0]
                iupred_results[wt_seq] = wt_idrpred
            else:
                wt_idrpred = iupred_results[wt_seq]
            row['wt_idrpred'] = wt_idrpred
            if wt_seq not in pscore_results.keys():
                wt_pipi_dict = position_scores(wt_seq)
                pscore_results[wt_seq] = wt_pipi_dict
            else:
                wt_pipi_dict = pscore_results[wt_seq]
            row['pi_dict'] = wt_pipi_dict
            row['pfam_info'] = seq_pfam_info[mt_seq]
            row['in_pfam'] = seq_pfam_mapped[mt_seq]
            row['dist2IDR'] = seq_pfam_dist[mt_seq]
            bf = buildFeatures()
            bf.call(row)
            return_dict = bf.return_dict
            return_dict['label'] = row['label']
            return_dict['Gene_Name'] = str(row['Gene_Name'])
            ml_df = ml_df.append(return_dict,ignore_index=True)
        ml_df.to_csv(f'data/{output_df_name}',sep='\t')

