import runpy



# runpy.run_path(path_name='evalute_psmutpred_performances/ml_feature_construction.py', run_name='__main__')

# runpy.run_path(path_name='/code/evalute_psmutpred_performances/build_background_dataset.py', run_name='__main__')

runpy.run_path(path_name='/code/evalute_psmutpred_performances/PSMutpred_evaluation.py')

runpy.run_path(path_name='evaluate_patho_prediction/patho_predict_with_ESM1b.py')
runpy.run_path(path_name='evaluate_patho_prediction/patho_predict_with_EVE.py')

print('excuting predict_variants_using_PSMutPred.py') ### example for prediction using PSMutPred
runpy.run_path(path_name='predict_variants_using_PSMutPred.py') ### example for prediction using PSMutPred
