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
confusionMatrix(y_test, pred, positive = "More.50k")

#train
# Prediction Less.50k More.50k
# Less.50k     4031      913
# More.50k      282     1286
    
  
#test
# Prediction Less.50k More.50k
# Less.50k     9821     2614
# More.50k      731     3115


##### Over sampling: 
pred2 = predict(model_rf_over, x_test)
confusionMatrix(y_test, pred2, positive = "More.50k")

#train
# Prediction Less.50k More.50k
# Less.50k     4404      540
# More.50k      473     1095

#test

# Prediction Less.50k More.50k
# Less.50k    10977     1458
# More.50k     1269     2577



#### Smote: 
pred3 = predict(model_rf_smote, x_test)
confusionMatrix(y_test, pred3, positive = "More.50k")

#train
# Prediction Less.50k More.50k
# Less.50k     4696      248
# More.50k      685      883
#test

# Prediction Less.50k More.50k
# Less.50k    11658      777
# More.50k     1606     2240


#########################################
# Use ROC metric
#########################################

# stratified sample with ROC 

pred4 = predict(rf_strata, x_test)
confusionMatrix(y_test, pred4, positive = "More.50k")
#train
# Prediction Less.50k More.50k
# Less.50k     3978      966
# More.50k      221     1347

#test
# Prediction Less.50k More.50k
# Less.50k     9782     2653
# More.50k      577     3269


# with down sample + ROC

pred5 = predict(down_fit, x_test)
confusionMatrix(y_test, pred5, positive = "More.50k")


#train
# Prediction Less.50k More.50k
# Less.50k     4016      928
# More.50k      307     1261
#test

# Prediction Less.50k More.50k
# Less.50k     9895     2540
# More.50k      712     3134

############################
# up sample with roc

pred6 = predict(up_fit, x_test)
confusionMatrix(y_test, pred6, positive = "More.50k")

#train
# Prediction Less.50k More.50k
# Less.50k     4286      658
# More.50k      366     1202

#test
# Prediction Less.50k More.50k
# Less.50k    10667     1768
# More.50k      976     2870


############################
# smote sample with roc

pred7 = predict(smote_fit, x_test)
confusionMatrix(y_test, pred7, positive = "More.50k")

#train
# Prediction Less.50k More.50k
# Less.50k     4603      341
# More.50k      619      949
#test
# Prediction Less.50k More.50k
# Less.50k    11393     1042
# More.50k     1536     2310


##################################################################################
#Model with weights: 
##################################################################################
# with weights

pred8 = predict(weighted_fit, x_test)
confusionMatrix(y_test, pred8, positive = "More.50k")

#train
# Prediction Less.50k More.50k
# Less.50k     4702      242
# More.50k      652      916
#test
# Prediction Less.50k More.50k
# Less.50k    11718      717
# More.50k     1616     2230

##############################
# with weight  and strata: 
pred9 = predict(weighted_strata, x_test)
confusionMatrix(y_test, pred9, positive = "More.50k")

#train

# Prediction Less.50k More.50k
# Less.50k     3950      994
# More.50k      226     1342
#test
# Prediction Less.50k More.50k
# Less.50k     9699     2736
# More.50k      573     3273

#weight with down samp: 
pred10 = predict(weighted_down_fit, x_test)
confusionMatrix(y_test, pred10, positive = "More.50k")

#train
# Prediction Less.50k More.50k
# Less.50k     4093      851
# More.50k      239     1329

#test
# Prediction Less.50k More.50k
# Less.50k    10033     2402
# More.50k      634     3212

pred11 = predict(weighted_down_fit2, x_test)
confusionMatrix(y_test, pred11, positive = "More.50k")

#train
# Prediction Less.50k More.50k
# Less.50k     4696      248
# More.50k      625      943

#test
# Prediction Less.50k More.50k
# Less.50k    11616      819
# More.50k     1572     2274


pred12= predict(weighted_smote_fit, x_test)
confusionMatrix(y_test, pred12, positive = "More.50k")


#train
# Prediction Less.50k More.50k
# Less.50k     4605      339
# More.50k      624      944
#test
# Prediction Less.50k More.50k
# Less.50k    11427     1008
# More.50k     1552     2294

##################################################################################
#Tunning the parameter with best strategy: 
##################################################################################

####### Tuning mtries and trees: 
##### random search: 

pred13= predict(rf_random, x_test)
confusionMatrix(y_test, pred13, positive = "More.50k")

#train:
# Prediction Less.50k More.50k
# Less.50k     4567      377
# More.50k      604      964

#test
# Prediction Less.50k More.50k
# Less.50k    11283     1152
# More.50k     1515     2331


##### grid search: 
pred14= predict(rf_gridsearch, x_test)
confusionMatrix(y_test, pred14, positive = "More.50k")



