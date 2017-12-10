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
x_test = df_test[,-c("income")]
y_test = df_test$income

#############################################
#Sampling technique for imbalanced class: 
#############################################
######Under sampling: 

pred = predict(model_rf_under, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
#train

#test

##### Over sampling: 
pred2 = predict(model_rf_over, xtest_origin)
confusionMatrix(ytest_origin, pred2, positive = "More.50k")
#train

#test



#### Smote: 
pred3 = predict(model_rf_smote, xtest_origin)
confusionMatrix(ytest_origin, pred3, positive = "More.50k")

#train

#test

#########################################
# Use ROC metric
#########################################

# stratified sample with ROC 

pred4 = predict(rf_strata, xtest_origin)
confusionMatrix(ytest_origin, pred4, positive = "More.50k")
#train

#test
# with down sample + ROC

pred5 = predict(down_fit, xtest_origin)
confusionMatrix(ytest_origin, pred5, positive = "More.50k")


#train

#test


############################
# up sample with roc

pred6 = predict(up_fit, xtest_origin)
confusionMatrix(ytest_origin, pred6, positive = "More.50k")

#train

#test


############################
# smote sample with roc

pred7 = predict(smote_fit, xtest_origin)
confusionMatrix(ytest_origin, pred7, positive = "More.50k")

#train
#test



##################################################################################
#Model with weights: 
##################################################################################
# with weights

pred8 = predict(weighted_fit, xtest_origin)
confusionMatrix(ytest_origin, pred8, positive = "More.50k")

#train

#test


##############################
# with weight  and strata: 
pred9 = predict(weighted_strata, xtest_origin)
confusionMatrix(ytest_origin, pred9, positive = "More.50k")
#train

#weight with down samp: 
pred10 = predict(weighted_down_fit, xtest_origin)
confusionMatrix(ytest_origin, pred10, positive = "More.50k")



pred11 = predict(weighted_down_fit2, xtest_origin)
confusionMatrix(ytest_origin, pred11, positive = "More.50k")




pred12= predict(weighted_smote_fit, xtest_origin)
confusionMatrix(ytest_origin, pred12, positive = "More.50k")




##################################################################################
#Tunning the parameter with best strategy: 
##################################################################################

####### Tuning mtries and trees: 
##### random search: 

pred13= predict(rf_random, xtest_origin)
confusionMatrix(ytest_origin, pred13, positive = "More.50k")



##### grid search: 
pred14= predict(rf_gridsearch, xtest_origin)
confusionMatrix(ytest_origin, pred14, positive = "More.50k")



