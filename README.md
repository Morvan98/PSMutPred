## News:
Our user friendly website is available:
Precomputed PSMutPred scores for variants from the Uniprot human-reviewed proteome are available at:  http://www.psmutpred.online/ .

## How to install:  

1. Install required packages via pip:
   Python 3.7 is recommanded. The required packages can be installed using pip:
```
### in command line
pip install \
biopython joblib==1.1.0 numpy pandas pickle5==0.0.11 scikit-learn scipy tqdm openpyxl
```
  or

  
2. By cloning the repository:
   Run the following command:
```
### in command line
git clone https://github.com/Morvan98/PSMutPred.git ## 1-2 minutes
cd PSMutPred
conda env create -f environment.yml ## 1-2 minutes
conda activate psmutpred
python predict_variants_using_PSMutPred.py
```
## How to predict:  

1. Edit the script: 'predict_variants_using_PSMutPred.py'; then
  Replace the path of 'data/dataset/EPS8_clinvar.tsv' with the path to your own variant file:
  It should be noticed that the wild type amino acid has to be matched with the input protein name (uniprot_entry)
```
### in .py file within the main directory
from model import *
pd.set_option('display.max_columns',None)
eps8_human = pd.read_csv('data/dataset/EPS8_clinvar.tsv',sep='\t')
example_df = eps8_human[['wt_aa','mt_aa','position']]
print(example_df.head(10))
df_ = predict_df(example_df,'EPS8_HUMAN')
print(df_)
```
2. Run the prediction script:
```
### in command line
python predict_variants_using_PSMutPred.py
```
3. Expected outputs:
   The output will include predicted impacts on phase separation:
    IP-score: Propensity to impact phase separation, ranging from 1 (most likely to impact) to 0 (not likely to impact).
    SP-score: Propensity to strengthen phase separation, ranging from 1 (likely to strengthen phase separation) to 0 (likely to weaken phase separation).

4. Coming updates:
   I will soon update the prediction function for optional sequences and mutations as current version require known human gene_name for input.
   
## Paper codes
1. Run the main script:
```
### in command line
python main.py ### can be edited to run the specific file
```
2. Source data for PSMutPred paper can be found at data/source_data
3. Dataset that already merge random background dataset are in data/dataset:
   'data_merged.tsv; data_merged_aa_weighted_background.tsv; data_merged_weighted_sampling.tsv'
