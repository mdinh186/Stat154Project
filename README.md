# Stat154Project

team_doglovers/
     
     * README.md (directory of all files)
     * data/ 
        + adult.data (original dataset file)
        + adult.test (original dataset file)
        + data-dictionary.txt (original dataset description)
        + df_impute_feat.rds (imputed training dataset with feature engineering, validation set)
        + def_impute_test.rds (test set with imputed NA)
        + df_impute.rds (imputed training dataset)
        + df_remove.rds (training dataset with removed NA)
        + df_remove_feat.rds (training set with removed NA with feature engineering)
        + test.rds (validation test set)
        + toptenfeat.rds (randomforest top ten variables set)
        + train.rds (validation training set)
    * Image/
        + Model_Building/
            - Bagged_final_ROC.png (ROC curve for final bagged model, validation set)
            - Bagged_varImp_down_fit.png (variable importance for selected bagged model)
            - Bagged_model_comparison.png (comparing accuracy statistics for bagged models across sampling techniques)
            - decision_cv.png (tuning parameter graphs for decision tree)
            - decision_plotcp.png (plotting cp parameter for decision tree)
            - Decision_ROC.png (ROC curve for final decision tree, validation set)
            - decision_tree.png (final decision tree map)
            - decision_varimp.png (variable importance for final decision tree)
            - varim_majority.png
            - varim_minority.png
        + Performance/
            - Baseline (baseline performance statistics for NA datasets)
            - rf_final_ROC.png (ROC curve for final randomforest)
            - roc_curve_test_top_ten_hyperparam_tuning_smote_samp.png (ROC curve on top ten for smote sample)
            - roc_curve_test_top_ten_sampling_w_roc_.png (ROC curve on top ten, roc metric)
            - roc_curve_test_weighted_sampling_wroc.png (ROC curve on test data, roc metric, weighted)
            - roc_curve_training_weighted_sampling_roc.png (ROC curve on training data, weighted with roc metric)
            - TEST_Bagged_ROC.png (ROC curve for final bagged tree, on test set)
            - TEST_Decision_ROC.png (ROC curve for final decision tree, on test set)
            - test_top_ten_sampling_roc_curve.png (top ten final randomforest model, ROC curve against test data)
        + PreProcessing/
            - agevsinc.png (training set age by income class)
            - capital-gainvsinc.png (training set capital gains by income class)
            - capital-lossvsinc.png (training set capital loss by income class)
            - corr_plot.png (correlation plot of all 15 predictors)
            - degreevsinc.png (training set education by income class)
            - education-numvsinc.png (training set education years by income class)
            - fnlwgtvsinc.png (training set finalweight by income class)
            - hours-per-weekvsinc.png (training set hours worked by income class)
            - missing_value.png (examining missing values, EDA)
    * process/ 
        + Bagged_tree.R (prelim exploring bagged model)
        + Bagged_Best.R (Bagged model selection)
        + Bagged_Best_Test.R (Final bagged model against test data)
        + data_preprocessing.R (preprocessing datasets with features)
        + Decision.R (prelim exploring decision trees)
        + Decision_Best_Feat.R (Decision tree model selection)
        + Decision_Best_Test.R (Final decision tree against test data)
        + Emily_EDA_cleaning.Rmd (prelim EDA, code)
        + Functions2CSV.Rmd (main functions that were run to clean and feature the data before models)
        + Project_EDA.pdf (prelim EDA, pdf)
        + Project_EDA.pdf (prelim EDA, code)
        + model_roc_plot.R (function to run for multiple ROCs)
        + RandomForest.R (exploring randomforest)
        + Report_EDA_Combined.Rmd (consolidated EDA analysis)
        + RF_Best_Feat_test.R (Final randomforest model against test data) 
        + RF_Best_Feat.R (Randomforest model selection)
        + RF_test.R (prelim exploring randomforest trees)
     * report/
        + Final_Report_12.11.17.docx (word format of final report)
        + Final_Report_12.11.17.pdf (PDF format of final report)
        + Final_Report.Rmd (Rmd file for report)
        
        
      