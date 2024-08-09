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
git clone https://github.com/Morvan98/PSMutPred.git ## very fast
cd PSMutPred
conda env create -f environment.yml ## 1-2 minutes
conda activate psmutpred
python predict_variants_using_PSMutPred.py
```
## How to predict:  

1. Edit the script: 'predict_variants_using_PSMutPred.py'; then
  Replace the path of 'data/dataset/EPS8_clinvar.tsv' with the path to your own variant file:
```
from model import *
pd.set_option('display.max_columns',None)
eps8_human = pd.read_csv('data/dataset/EPS8_clinvar.tsv',sep='\t')
example_df = eps8_human[['wt_aa','mt_aa','position']]
print(example_df.head(10))
df_ = predict_df(example_df,'EPS8_HUMAN')
print(df_)
### in .py file
```
2. Run the prediction script:
```
python predict_variants_using_PSMutPred.py
### in command line
```
3. Expected outputs:
   The output will include predicted impacts on phase separation:
    IP-score: Propensity to impact phase separation, ranging from 1 (most likely to impact) to 0 (not likely to impact).
    SP-score: Propensity to strengthen phase separation, ranging from 1 (likely to strengthen phase separation) to 0 (likely to weaken phase separation).
## Reproduce necessary results
1. Run the main script:
```
python main.py
### in command line
```
