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


#############################################################
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
set.seed(123)


df_impute = readRDS("data/df_impute_feat.rds")
df_impute = data.table(df_impute)



#############################################################
# Prepocessing data:
#############################################################
## stratified sampling data:
# split1 = createDataPartition(df_impute$income, p  = 0.8)[[1]]
# 
# train_origin = df_impute[split1,]
# test_origin = df_impute[-split1, ]
# xtrain_origin = train_origin[, -c("income")]
# ytrain_origin = train_origin$income
# xtest_origin = test_origin[,-c("income")]
# ytest_origin = test_origin$income


train_origin = readRDS("data/train.rds")
xtrain_origin = train_origin[,-c("income")]
ytrain_origin = train_origin$income
#######################################
#make baseline models and feature sections


control = trainControl(method = "cv", number = 5)
rf_default  = train(x = xtrain_origin, y = ytrain_origin,
                  method = "rf", trControl = control, 
                    importance= T, ntree = 100)
rf_default
pred = predict(rf_default, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")
varImp(rf_default)



########################################################
### result for version1:  12/5 
####impute with feature: 
#### Accuracy: train: 86 %, test = 86%, t
### evaluation:

# Prediction Less.50k More.50k
# Less.50k     4634      310
# More.50k      591      977


#### impute without feature: 
#### Accuracy: train:82.7?%

# Reference
# Prediction  <=50K  >50K
# <=50K   4934    25
# >50K    1111   442


### remove missing value without feature: 
#### accuracy train: 81.7%

# Reference
# Prediction  <=50K  >50K
# <=50K   4529    22
# >50K    1022   459

### remove mssing values with feature: 
#### accuracy train:  82.8%
# Reference
# Prediction      Less.than.50k More.than.50k
# Less.than.50k          4473            32
# More.than.50k          1065           462


########################################################
### result for version 2:  12/5 



#############################################
#Sampling technique for imbalanced class: 
#############################################
######Under sampling: 


ctrol = trainControl(method = "cv", number =5, 
                     verboseIter = F,
                     sampling = "down")

model_rf_under = train(xtrain_origin, ytrain_origin, method = "rf",
                       trControl = ctrol)

pred = predict(model_rf_under, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4290       58
# More.50k      654     1510
# Sensitivity : 0.9630          
# Specificity : 0.8677
# FPR : 0.13
# Area under the curve: 0.9699
##### Over sampling: 
ctrol2 = trainControl(method = "cv", number =5, 
                     verboseIter = F,
                     sampling = "up")
model_rf_over = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol2)


pred = predict(model_rf_over, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")

#
# Prediction Less.50k More.50k
# Less.50k     4824       98
# More.50k      120     1470
# Accuracy : 0.9665
# Sensitivity : 0.9375          
# Specificity : 0.9757
# FPR : 0.024
# Area under the curve: 0.9881



#### Smote: 
ctrol4 = trainControl(method = "cv", number =5, 
                               verboseIter = F,
                               sampling = "smote")

model_rf_smote = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol4)

pred = predict(model_rf_smote, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4761      568
# More.50k      183     1000
# Sensitivity : 0.6378          
# Specificity : 0.9630 
# Area under the curve: 0.9404

colnames(test_origin)[17] = "income"

model_list_samp = list(under_sampling = model_rf_under,
                       over_sampling = model_rf_over,
                       smote_sampling = model_rf_smote)
custom_col = c("#000000", "#009E73", "#0072B2")
model_roc_plot(model_list_samp, custom_col, AUC = T)



#########################################
# Use ROC metric
#########################################

# stratified sample with ROC 
#### 
control5 = trainControl(method = "cv", number = 5, search = "random",allowParallel = TRUE, 
                        summaryFunction = twoClassSummary, 
                        classProbs = T)

rf_strata = train(xtrain_origin, ytrain_origin, method = "rf", 
                  tuneLength = 15, trControl=control5, 
                  strata = ytrain_origin, sampsize = c(50,50), 
                  metric = "ROC")


pred = predict(rf_strata, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4013      238
# More.50k      931     1330
# Accuracy : 0.8205 
#Area under the curve: 0.9079
# with down sample + ROC

control5$sampling = "down"
down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                          verbose = F, metric = "ROC", 
                 trControl = control5)

pred = predict(down_fit, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4286       63
# More.50k      658     1505
# Accuracy : 0.8893  
# Sensitivity : 0.9598          
# Specificity : 0.8669          
# Area under the curve: 0.9679
############################
# up sample with roc
control5$sampling = "up"
up_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                        verbose = F, metric = "ROC", 
               trControl = control5)

pred = predict(up_fit, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")


# Prediction Less.50k More.50k
# Less.50k     4847      100
# More.50k       97     1467
# Accuracy : 0.8893  
# Sensitivity : 0.9598          
# Specificity : 0.8669 
#Area under the curve: 0.9882

############################
# smote sample with roc

control5$sampling = "smote"
smote_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                           verbose = F, metric = "ROC", 
                  trControl = control5)

pred = predict(smote_fit, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")


# Prediction Less.50k More.50k
# Less.50k     4770      163
# More.50k      174     1405
#Accuracy : 0.9482
# Sensitivity : 0.8960          
# Specificity : 0.9648
#Area under the curve: 0.9806

model_list_roc_samp = list(under_sampling_w_roc = down_fit, 
                           over_sampling_w_roc= up_fit, 
                           smote_sampling_w_roc = smote_fit, 
                           stratified_50_sampling_w_roc = rf_strata)
custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00")
model_roc_plot(model_list_roc_samp, custom_col, AUC = T)



##################################################################################
#Model with weights: 
##################################################################################
# with weights
model_weights = ifelse(ytrain_origin == "Less.50k", 
                       (1/table(ytrain_origin)[1]) * 0.5,
                       (1/table(ytrain_origin)[2]) * 0.5)

ctrol8 = trainControl(method = "cv", number =5, 
                      summaryFunction = twoClassSummary, 
                      classProbs = T)

weighted_fit = train(xtrain_origin, ytrain_origin, method = "rf",
                     verbose = F,
                     weights = model_weights, 
                     metric = "ROC",
                     trControl = ctrol8)
pred = predict(weighted_fit, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4882      134
# More.50k       62     1438
# Sensitivity : 0.9145          
# Specificity : 0.9875 

##############################
# with weight  and strata: 

weighted_strata = train(xtrain_origin, ytrain_origin, method = "rf", 
      tuneLength = 13, trControl=ctrol8, 
      strata = ytrain_origin, 
      sampsize = c(50,50), 
      metric = "ROC",
      weights = model_weights)

pred = predict(weighted_strata, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")


# Prediction Less.50k More.50k
# Less.50k     3991      230
# More.50k      953     1338
# Accuracy : 0.8183 
# Sensitivity : 0.8533          
# Specificity : 0.8072 
#Area under the curve: 0.9154

ctrol8 = trainControl(method = "cv", number =5, 
                      summaryFunction = twoClassSummary, 
                      classProbs = T)
nmin = min(table(ytrain_origin))
weighted_down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                        tuneLength = 15, trControl=ctrol8, 
                        strata = ytrain_origin, 
                        sampsize = rep(nmin, 2), metric = "ROC",
                        weights = model_weights)

pred = predict(weighted_down_fit, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4355       86
# More.50k      589     1482
# Accuracy : 0.8963 
# Sensitivity : 0.9452          
# Specificity : 0.8809
# Area under the curve: 0.9771
nmin = min(table(ytrain_origin))
weighted_down_fit2 = train(xtrain_origin, ytrain_origin, method = "rf", 
                          tuneLength = 15, trControl=ctrol8, 
                          strata = ytrain_origin, 
                          metric = "ROC",
                          replace = F,
                          weights = model_weights)




pred = predict(weighted_down_fit2, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")

# Confusion Matrix and Statistics

# Prediction Less.50k More.50k
# Less.50k     4811      277
# More.50k      133     1291
# Accuracy : 0.937
# Sensitivity : 0.8233          
# Specificity : 0.9731
#Area under the curve: 0.9174

ctrol8$sampling = "smote"
weighted_smote_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                        tuneLength = 15, 
                        trControl=ctrol8, 
                        strata = ytrain_origin, 
                        metric = "ROC",
                        weights = model_weights)

pred = predict(weighted_smote_fit, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k     4792      250
# More.50k      152     1318
# Accuracy : 0.9383
# Sensitivity : 0.8406         
# Specificity : 0.9693
# Area under the curve: 0.9114
model_list_weight_samp = list(
                              weighted = weighted_fit2, 
                              weighted_down_samp_w_roc= weighted_down_fit2, 
                              weighted_smote_sampling_w_roc = weighted_smote_fit2, 
                              weighted_stratified_sampling_w_roc = weighted_down_fit22)
custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00")
model_roc_plot(model_list_weight_samp, custom_col, AUC = T)



############################################
# tuning parameter:


rf_random = train(xtrain_origin, ytrain_origin, method = "rf", 
                  verbose = F, metric = "ROC", tuneLength = 2:10,
                  trControl = control5)
###rf_random results:

pred = predict(rf_random, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")



##### grid search: 

customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

control5$search = "grid"
tunegrid = expand.grid(.mtry=c(2:9), .ntree=seq(50,300,50))

rf_gridsearch = train(xtrain_origin, ytrain_origin, method = customRF, 
                      verbose = F, metric = "ROC", tuneGrid = tunegrid,
                      trControl = control5)

pred = predict(rf_gridsearch, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")


# Prediction Less.50k More.50k
# Less.50k     4602      342
# More.50k      578      990


model_list_tuning = list(baseline = rf_default,
                         random_search_smote= rf_random, 
                         grid_search_smote = rf_gridsearch)
custom_col = c("#000000", "#009E73", "#0072B2")
model_roc_plot(model_list_tuning, custom_col, AUC = T)


########################################################################
#compare accross decision tree, tree bags, and random forest 

fiveStats = function(...) c(twoClassSummary(...), defaultSummary(...))
fourStats = function(data, lev = levels(data$obs), model = NULL){
  accKapp = postResample(data[, "pred"], data[, "obs"])
  out = c(accKapp, sensitivity(data[,"pred"], data[, "obs"], lev[1]), specificity(data[, "pred"], data[, "obs"], lev[2]))
  names(out)[3:4] = c('Sens', "Spec")
  out
}
control = trainControl(method = "cv", number = 5, classProbs = T)
ctrlNoProb = control
ctrlNoProb$classProbs = F
ctrlNoProb$summaryFunction  = fourStats



rfFit = train(xtrain_origin, ytrain_origin, method = "rf", 
              trControl = control, ntree = 100, 
               metric = "ROC" )
rfFit_treebag = train(xtrain_origin, ytrain_origin, method = "treebag", 
              trControl = control, ntree = 100, 
              metric = "ROC")
rfFit_dt = train(xtrain_origin, ytrain_origin, method = "treebag", 
                      trControl = control, 
                      metric = "ROC")

evalResults = data.frame( income = evaluation$income ) # put in the truth
evalResults$RF = predict( rfFit, newdata=evaluation, type="prob" )[,1]
evalResults$treebag= predict( rfFit_treebag, newdata=evaluation, type="prob" )[,1]
evalResults$dt= predict(rfFit_dt, newdata=evaluation, type="prob" )[,1]

rfROC = roc( evalResults$income, evalResults$RF, levels:rev( levels(evalResults$income) ) )
treebagOC = roc( evalResults$income, evalResults$treebag, levels=rev( levels(evalResults$income) ) )
dtROC = roc( evalResults$income, evalResults$dt, levels=rev( levels(evalResults$income) ) )




##################################################################################
# try cost sensitive metrics:  
##################################################################################


