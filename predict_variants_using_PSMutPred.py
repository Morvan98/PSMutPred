'''
example for prediction using PSMutPred
input: 
    dataframe containing variants, four rows are required
    gene_name, mt_aa, wt_aa, position
output:
    dataframe of PSMutPred scores and rank scores
'''
from model import *
pd.set_option('display.max_columns',None)
eps8_human = pd.read_csv('data/dataset/EPS8_clinvar.tsv',sep='\t')
example_df = eps8_human[['wt_aa','mt_aa','position']]
print(example_df.head(10))

df_ = predict_df(example_df,'EPS8_HUMAN')

print(df_)