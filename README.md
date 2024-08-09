# How to install: 
1. Install packages by pip and run files
  Required python packages (python 3.7 is required; packages can be installed by pip):
  biopython; joblib==1.1.0; numpy; pandas; pickle5==0.0.11; scikit-learn; scipy; tqdm; openpyxl
2. Install by git:
   Running:
   ```
   git clone https://github.com/Morvan98/PSMutPred.git
   ```
# How to predict:
1.Edit the script file: 'predict_variants_using_PSMutPred.py'
  replace the path of 'data/dataset/EPS8_clinvar.tsv' by your own variant file
  ```
  from model import *
  pd.set_option('display.max_columns',None)
  eps8_human = pd.read_csv('data/dataset/EPS8_clinvar.tsv',sep='\t')
  example_df = eps8_human[['wt_aa','mt_aa','position']]
  print(example_df.head(10))
  
  df_ = predict_df(example_df,'EPS8_HUMAN')
  
  print(df_)
  ```
2. run 
```
python predict_variants_using_PSMutPred.py
```
    
