library(rpart)
library(randomForest)
library(data.table)
library(caret)
library(doParallel)
library(plotROC)
library(dplyr)
library(purrr)
library(pROC)
library(grid)


df_test = readRDS("data/df_impute_test.rds")
# df_test = data.table(df_test)
# df_test$income = as.character(df_test$income)
# df_test[, income := ifelse(income == " <=50K.", "Less.50k", "More.50k")]
# char_vec = c("workclass", "marital_status","occupation","relationship","native_country","gen_race", "income")
# df_test[, (char_vec) := lapply(.SD, function (x) as.factor(x)), .SDcols = char_vec]
#saveRDS(df_test, "data/df_impute_test.rds")


x_test = df_test[,..top_ten]
y_test = df_test$income



#############################################
#Sampling technique for imbalanced class: 
#############################################
######Under sampling: 

pred = predict(model_rf_under, x_test)
confusionMatrix(pred, y_test,  positive = "More.50k")


  
#test
# Prediction Less.50k More.50k
# Less.50k     9822      729
# More.50k     2613     3117
# Accuracy : 0.7945 
# Sensitivity : 0.8105          
# Specificity : 0.7899  
# FPR : 0.21
#Area under the curve: 0.9021

##### Over sampling: 
pred2 = predict(model_rf_over, x_test)
confusionMatrix(pred2, y_test, positive = "More.50k")



#test

# Prediction Less.50k More.50k
# Less.50k    10973     1270
# More.50k     1462     2576
#Accuracy : 0.8325 
# Sensitivity : 0.6698          
# Specificity : 0.8824 
# FPR: 0.11
# Area under the curve: 0.9019

#### Smote: 
pred3 = predict(model_rf_smote, x_test)
confusionMatrix(pred3, y_test,  positive = "More.50k")


#test

# Prediction Less.50k More.50k
# Less.50k    11658     1605
# More.50k      777     2241
# Accuracy : 0.8537 
# Sensitivity : 0.5827          
# Specificity : 0.9375   
# FPR : 0.06
#Area under the curve: 0.8882

#########################################
# Use ROC metric
#########################################

# stratified sample with ROC 

pred4 = predict(rf_strata, x_test)
confusionMatrix(pred4, y_test, positive = "More.50k")


#test
# Prediction Less.50k More.50k
# Less.50k     9784      575
# More.50k     2651     3271
# Accuracy : 0.8019 
# Sensitivity : 0.8505         
# Specificity : 0.7868 
# FPR : 0.21
#Area under the curve: 0.9076

# with down sample + ROC

pred5 = predict(down_fit, x_test)
confusionMatrix(pred5, y_test,  positive = "More.50k")


#test

# Prediction Less.50k More.50k
# Less.50k     9896      712
# More.50k     2539     3134

# Accuracy : 0.8001  
# Sensitivity : 0.8149          
# Specificity : 0.7958
# FPR: 0.20 
# Area under the curve: 0.8989
############################
# up sample with roc

pred6 = predict(up_fit, x_test)
confusionMatrix(pred6,y_test, positive = "More.50k")

#train
# Prediction Less.50k More.50k
# Less.50k     4286      658
# More.50k      366     1202

#test
# Prediction Less.50k More.50k
# Less.50k    10667      975
# More.50k     1768     2871
# Accuracy : 0.8315
# Sensitivity : 0.7465          
# Specificity : 0.8578 
# FPR: 0.14
#Area under the curve: 0.9073
############################
# smote sample with roc

pred7 = predict(smote_fit, x_test)
confusionMatrix(pred7, y_test, positive = "More.50k")

#train
# Prediction Less.50k More.50k
# Less.50k     4603      341
# More.50k      619      949
#test
# Prediction Less.50k More.50k
# Less.50k    11395     1535
# More.50k     1040     2311
# Accuracy : 0.8417  
# Sensitivity : 0.6009          
# Specificity : 0.9164 
# FPR: 0.08
#Area under the curve: 0.8947
#
##################################################################################
#Model with weights: 
##################################################################################
# with weights
pred8 = predict(weighted_fit, x_test)
confusionMatrix(pred8, y_test, positive = "More.50k")


# Prediction Less.50k More.50k
# Less.50k    11718     1616
# More.50k      717     2230
# Accuracy : 0.8567 
# Sensitivity : 0.5798          
# Specificity : 0.9423
# FPR: 0.05
#Area under the curve: 0.877
##############################
# with weight  and strata: 
pred9 = predict(weighted_strata, x_test)
confusionMatrix(pred9, y_test,   positive = "More.50k")


#test
# Prediction Less.50k More.50k
# Less.50k     9699      574
# More.50k     2736     3272

# Accuracy : 0.7968
# Sensitivity : 0.8508          
# Specificity : 0.7800 
# FPR: 0.22
#Area under the curve: 0.9056


#weight with down samp: 
pred10 = predict(weighted_down_fit, x_test)
confusionMatrix(pred10, y_test, positive = "More.50k")

test_origin = readRDS("data/test.rds")
x_test = test_origin[,-c("income")]
y_test = test_origin$income

#test
# Prediction Less.50k More.50k
# Less.50k    10032      636
# More.50k     2403     3210
# Accuracy : 0.8135  
# Sensitivity : 0.8346          
# Specificity : 0.8068 
# FPR: 0.19
# Area under the curve: 0.9137

pred11 = predict(weighted_down_fit2, x_test)
confusionMatrix(pred11, y_test,  positive = "More.50k")



#test
# Prediction Less.50k More.50k
# Less.50k    11617     1574
# More.50k      818     2272
# Accuracy : 0.8531  
# Sensitivity : 0.5907          
# Specificity : 0.9342
# FPR : 0.06

pred12= predict(weighted_smote_fit, x_test)
confusionMatrix(pred12, y_test,   positive = "More.50k")


#test 
# Prediction Less.50k More.50k
# Less.50k    11427     1551
# More.50k     1008     2295
# Accuracy : 0.8428
# Sensitivity : 0.5967          
# Specificity : 0.9189     
# FPR : 0.08
#Area under the curve: 0.896
##################################################################################
#Tunning the parameter with best strategy: 
##################################################################################

####### Tuning mtries and trees: 
##### random search: 

pred13= predict(rf_random, x_test)
confusionMatrix(pred13, y_test, positive = "More.50k")


#test
# Prediction Less.50k More.50k
# Less.50k    11279     1514
# More.50k     1156     2332
# Accuracy : 0.8362
# Sensitivity : 0.6063          
# Specificity : 0.9070 
# FPR : 0.09
#Area under the curve: 0.895

##### grid search: 
pred14= predict(rf_gridsearch, x_test)
confusionMatrix(pred14, y_test,   positive = "More.50k")


# Prediction Less.50k More.50k
# Less.50k    11375     1484
# More.50k     1060     2362
# Accuracy : 0.8439
# Sensitivity : 0.6141          
# Specificity : 0.9148
# FPR : 0.08
#Area under the curve: 0.8927