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

pred = predict(model_rf_under, x_test)
confusionMatrix(pred, y_test, positive = "More.50k")


#test
# Prediction Less.50k More.50k
# Less.50k     9965      632
# More.50k     2470     3214
# Accuracy : 0.8095 
# Sensitivity : 0.8357          
# Specificity : 0.8014
#Area under the curve: 0.9034
#independent group: 
# Prediction Less.50k More.50k
# Less.50k     9838      650
# More.50k     2597     3196
#Accuracy : 0.8006 
# Sensitivity : 0.8310          
# Specificity : 0.7912
# FPR: 0.208
#Area under the curve: 0.897


##### Over sampling: 
pred2 = predict(model_rf_over2, x_test)
confusionMatrix(pred2, y_test, positive = "More.50k")


#test
# Prediction Less.50k More.50k
# Less.50k    11142     1293
# More.50k     1226     2620
# Accuracy : 0.8453 
# Sensitivity : 0.6696 
# FPR: 0.10
#Area under the curve: 0.9005
#IG:

# Prediction Less.50k More.50k
# Less.50k    11101     1233
# More.50k     1334     2613
# Sensitivity : 0.6794          
# Specificity : 0.8927 
# FPR: 0.1
#Area under the curve: 0.8995

#### Smote: 
pred3 = predict(model_rf_smote2, x_test)
confusionMatrix(pred3, y_test, positive = "More.50k")


#test
# Prediction Less.50k More.50k
# Less.50k    11776      659
# More.50k     1637     2209
# Accuracy : 0.859 
#Sensitivity : 0.7702 
# FPR: 0.05
#Area under the curve: 0.8877

#
# Prediction Less.50k More.50k
# Less.50k    11393     1428
# More.50k     1042     2418
#Accuracy : 0.8483
# Sensitivity : 0.6287          
# Specificity : 0.9162
#Area under the curve: 0.8966
#########################################
# Use ROC metric
#########################################

# stratified sample with ROC 

pred4 = predict(rf_strata2, x_test)
confusionMatrix(pred4, y_test, positive = "More.50k")


#test
# Prediction Less.50k More.50k
# Less.50k     9813     2622
# More.50k      609     3237
# Accuracy : 0.8015  
# Sensitivity : 0.5525 
# FPR: 0.21
# with down sample + ROC

pred5 = predict(down_fit2, x_test)
confusionMatrix(pred5, y_test, positive = "More.50k")

#Area under the curve: 0.9014

############################
# up sample with roc

pred6 = predict(up_fit2, x_test)
confusionMatrix(pred6, y_test, positive = "More.50k")


#test


# Prediction Less.50k More.50k
# Less.50k     9851      549
# More.50k     2584     3297
# Accuracy : 0.8076
# Sensitivity : 0.8573          
# Specificity : 0.7922   
# FPR : 0.20
# Area under the curve: 0.8967
# IG:
#   Prediction Less.50k More.50k
# Less.50k    10468      762
# More.50k     1967     3084
# Sensitivity : 0.8019          
# Specificity : 0.8418 
# Accuracy : 0.8324
# FPR : 0.15


############################
# smote sample with roc

pred7 = predict(smote_fit2, x_test)
confusionMatrix(pred7, y_test, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k    11299     1338
# More.50k     1136     2508
# Accuracy : 0.848    
# Sensitivity : 0.6521          
# Specificity : 0.9086
# FPR : 0.09
#Area under the curve: 0.8993
#IG: 
# Prediction Less.50k More.50k
# Less.50k    11654     1571
# More.50k      781     2275
#Sensitivity : 0.5915        
#Specificity : 0.9372
# FPR : 0.06

##################################################################################
#Model with weights: 
##################################################################################
# with weights

pred8 = predict(weighted_fit2, x_test)
confusionMatrix(pred8, y_test, positive = "More.50k")


#test: 
# Prediction Less.50k More.50k
# Less.50k    11507     1467
# More.50k      928     2379
# Sensitivity : 0.6186          
# Specificity : 0.9254
# FPR : 0.07


# IG: 
# Prediction Less.50k More.50k
# Less.50k    11614     1565
# More.50k      821     2281
# Accuracy : 0.8534 
# Sensitivity : 0.5931          
# Specificity : 0.9340
# FPR : 0.06
#Area under the curve: 0.9021
##############################
# with weight  and strata: 
pred9 = predict(weighted_strata2, x_test)
confusionMatrix(pred9, y_test, positive = "More.50k")

#
# Prediction Less.50k More.50k
# Less.50k     9759      571
# More.50k     2676     3275
# Accuracy : 0.8006
# Sensitivity : 0.8515          
# Specificity : 0.7848
# FPR : 21.5% 



#weight with down samp: 
pred10 = predict(weighted_down_fit, x_test)
confusionMatrix(pred10, y_test, positive = "More.50k")
#
# Prediction Less.50k More.50k
# Less.50k    10234      681
# More.50k     2201     3165
# Accuracy : 0.823    
# Sensitivity : 0.8229         
# Specificity : 0.8230 
# FPR : 0.17
#Area under the curve: 0.9095

pred11 = predict(weighted_down_fit2, x_test)
confusionMatrix(pred11, y_test, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k    11600     1433
# More.50k      835     2413
# Accuracy : 0.8607
# Sensitivity : 0.6274         
# Specificity : 0.9329  
# Area under the curve: 0.9045

pred12= predict(weighted_smote_fit, x_test)
confusionMatrix(pred12, y_test, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k    11446     1393
# More.50k      989     2453
# Accuracy : 0.8537 
# Sensitivity : 0.6378          
# Specificity : 0.9205 
# FPR: 0.07
# Area under the curve: 0.8988
##################################################################################
#Tunning the parameter with best strategy: 
##################################################################################

####### Tuning mtries and trees: 
##### random search: 

pred13= predict(rf_random, x_test)
confusionMatrix(pred13, y_test, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k    11698     1558
# More.50k      737     2288
# Accuracy : 0.859
# Sensitivity : 0.5949          
# Specificity : 0.9407 
# FPR: 

##### grid search: 
pred14= predict(rf_gridsearch, x_test)
confusionMatrix(pred14, y_test, positive = "More.50k")


# Prediction Less.50k More.50k
# Less.50k    11433     1419
# More.50k     1002     2427


