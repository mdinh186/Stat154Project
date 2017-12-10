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


#############################################################
cl = makeCluster(detectCores()-1)
registerDoParallel(cl)
#############################################################

df_impute = readRDS("data/df_impute_feat.rds")
df_impute = data.table(df_impute)

#saveRDS(df_impute,"data/df_impute_feat.rds")


#############################################################
# Prepocessing data:
#############################################################
## stratified sampling data:
split1 = createDataPartition(df_impute$income, p  = 0.8)[[1]]

train_origin = df_impute[split1,]
test_origin = df_impute[-split1, ]
xtrain_origin = train_origin[, -c("income")]
ytrain_origin = train_origin$income
xtest_origin = test_origin[,-c("income")]
ytest_origin = test_origin$income


#######################################
#make baseline models and feature sections


control = trainControl(method = "cv", number = 5)
rf_default  = train(x = xtrain_origin, y = ytrain_origin,
                  method = "rf", trControl = control, 
                    importance= T, ntree = 100)
rf_default
pred = predict(rf_default, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
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
# (use AUC as metrics)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction  <=50K  >50K
# <=50K   4602   322
# >50K     589   999

# Reference
# Prediction Less.50k More.50k
# Less.50k     4571      377
# More.50k      567      997

# Reference
# Prediction Less.50k More.50k
# Less.50k     4571      377
# More.50k      572      992

##########################################################################################################################
# Random Forest Parameter tuning: 




####### Tuning mtries: 
##### random search: 


control = trainControl(method = "cv", number = 5, search = "random",allowParallel = TRUE)
mtry = sqrt(ncol(xtrain_origin))
rf_random = train(xtrain_origin, ytrain_origin, method = "parRF", 
                  tuneLength = 15, trControl=control)

###rf_random results:

      # The final value used for the model was mtry = 4.
      
    
pred = predict(rf_random, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")



##### grid search: 
control2 = trainControl(method = "cv", number = 5, search = "grid", allowParallel = TRUE)
tunegrid = expand.grid(.mtry = c(4:7))
rf_gridsearch = train(xtrain_origin, ytrain_origin, method = "rf",
                      tuneGrid=tunegrid, trControl=control2)

###rf_gridsearch results

        # The final value used for the model was mtry = 4.

pred_grid = predict(rf_random, xtest_origin)
confusionMatrix(ytest_origin, pred_grid, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k     4600      337
# More.50k      568     1007 



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
confusionMatrix(ytest_origin, pred, positive = "More.50k")



# Prediction Less.50k More.50k
# Less.50k     4077      867
# More.50k      240     1328


##### Over sampling: 
ctrol2 = trainControl(method = "cv", number =5, 
                     verboseIter = F,
                     sampling = "up")
model_rf_over = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol2)


pred = predict(model_rf_over, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")



# Reference
# Prediction Less.50k More.50k
# Less.50k     4474      470
# More.50k      477     1091



#### Smote: 
ctrol4 = trainControl(method = "cv", number =5, 
                               verboseIter = F,
                               sampling = "smote")

model_rf_smote = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol4)

pred = predict(model_rf_smote, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")



# Prediction Less.50k More.50k
# Less.50k     4730      214
# More.50k      684      884



model_list1 = list(original = rf_default, 
                   down =  model_rf_under, 
                   up = model_rf_over, 
                   smote = model_rf_smote)

custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00")

model_roc_plot(model_list1, custom_col)



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
confusionMatrix(ytest_origin, pred, positive = "More.50k")

# Reference
# Prediction Less.50k More.50k
# Less.50k     3994      950
# More.50k      223     1345


# with down sample + ROC

control5$sampling = "down"
down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                          verbose = F, metric = "ROC", 
                 trControl = control5)

pred = predict(down_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4059      885
# More.50k      209     1359


############################
# up sample with roc
control5$sampling = "up"
up_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                        verbose = F, metric = "ROC", 
               trControl = control5)

pred = predict(up_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


# 
# Prediction Less.50k More.50k
# Less.50k     4239      705
# More.50k      272     1296

# TPR = .647
# FPR= 0.06
# TNR = 0.93
# FNR: 0.35

# Less.50k     4203      741
# More.50k      251     1317
# 
# TPR: .639
# FPR: 0.05
# TNR: 0.94
# FNR: 0.36
############################
# smote sample with roc

control5$sampling = "smote"
smote_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                           verbose = F, metric = "ROC", 
                  trControl = control5)

pred = predict(smote_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


# 
# Reference
# Prediction Less.50k More.50k
# Less.50k     4672      272
# More.50k      632      936



model_list2 = list(original = rf_default,
                   strata = rf_strata,
                   down_roc = down_fit,
                   up_roc = up_fit,
                   SMOTE_roc = smote_fit)


custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
model_roc_plot(model_list2, custom_col)





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
confusionMatrix(ytest_origin, pred, positive = "More.50k")

# Reference
# Prediction Less.50k More.50k
# Less.50k     4628      316
# More.50k      601      967



##############################
# with weight  and strata: 

weighted_strata = train(xtrain_origin, ytrain_origin, method = "rf", 
      tuneLength = 13, trControl=ctrol8, 
      strata = ytrain_origin, 
      sampsize = c(50,50), 
      metric = "ROC",
      weights = model_weights)

pred = predict(weighted_strata, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


# Reference
# Prediction Less.50k More.50k
# Less.50k     3979      965
# More.50k      212     1356

nmin = min(table(ytrain_origin))
weighted_down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                        tuneLength = 15, trControl=ctrol8, 
                        strata = ytrain_origin, 
                        sampsize = rep(nmin, 2), metric = "ROC",
                        weights = model_weights)

pred = predict(weighted_down_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")

# 
# Prediction Less.50k More.50k
# Less.50k     4203      741
# More.50k      251     1317



nmin = min(table(ytrain_origin))
weighted_down_fit2 = train(xtrain_origin, ytrain_origin, method = "rf", 
                          tuneLength = 15, trControl=ctrol8, 
                          strata = ytrain_origin, 
                          metric = "ROC",
                          replace = F,
                          weights = model_weights)

pred = predict(weighted_down_fit2, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")

# Confusion Matrix and Statistics

# Prediction Less.50k More.50k
# Less.50k     4692      252
# More.50k      602      966





ctrol8$sampling = "smote"
weighted_smote_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                        tuneLength = 15, 
                        trControl=ctrol8, 
                        strata = ytrain_origin, 
                        metric = "ROC",
                        weights = model_weights)

pred = predict(weighted_smote_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4607      337
# More.50k      577      991



model_list4 = list(weighted = weighted_fit,
                   weighted_strata = weighted_strata,
                   down_weight = weighted_down_fit,
                   down_wight_wt_rp = weighted_down_fit2,
                   SMOTE_weight = weighted_smote_fit)


custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
model_roc_plot(model_list4, custom_col)



##################################################################################
#Tunning the parameter with best strategy: 
##################################################################################

####### Tuning mtries and trees: 
##### random search: 


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
# rerun the best model with best features: 
##################################################################################
up_fit_imp = data.frame(feature = rownames(varImp(up_fit)$importance), Score= varImp(up_fit)$importance)
up_fit_imp = up_fit_imp[order(up_fit_imp[,2], decreasing = T),]$feature
top_ten = c(as.character(up_fit_imp[1:10]), "income")


df_final = df_impute[,..top_ten]
train_final_idx = createDataPartition(df_final$income, p = 0.8)[[1]]
train_final = df_final[train_final_idx,]
test_final = df_final[-train_final_idx, ]
up_fit_feat = train(train_final[,-c("income")], train_final$income, method = "rf", 
               verbose = F, metric = "ROC", 
               trControl = control5)

# Prediction Less.50k More.50k
# Less.50k     4758      186
# More.50k      443     1125

pred = predict(up_fit_feat, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")

############################################
# tuning parameter:
control5$search = "random"

rf_random = train(xtrain_origin, ytrain_origin, method = "parRF", 
                  tuneLength = 15, trControl=control)

###rf_random results:




pred = predict(rf_random, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")



##### grid search: 
control2 = trainControl(method = "cv", number = 5, search = "grid", allowParallel = TRUE)
tunegrid = expand.grid(.mtry = c(4:7))

##################################################################################
# try cost sensitive metrics:  
##################################################################################
