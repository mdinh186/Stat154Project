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
# Less.50k     4290      654
# More.50k       59     1509


##### Over sampling: 
ctrol2 = trainControl(method = "cv", number =5, 
                     verboseIter = F,
                     sampling = "up")
model_rf_over = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol2)


pred = predict(model_rf_over, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k     4824      120
# More.50k       97     1471





#### Smote: 
ctrol4 = trainControl(method = "cv", number =5, 
                               verboseIter = F,
                               sampling = "smote")

model_rf_smote = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol4)

pred = predict(model_rf_smote, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k     4761      183
# More.50k      568     1000





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

# Prediction Less.50k More.50k
# Less.50k     4013      931
# More.50k      238     1330


# with down sample + ROC

control5$sampling = "down"
down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                          verbose = F, metric = "ROC", 
                 trControl = control5)

pred = predict(down_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4288      656
# More.50k       64     1504



############################
# up sample with roc
control5$sampling = "up"
up_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                        verbose = F, metric = "ROC", 
               trControl = control5)

pred = predict(up_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


# Prediction Less.50k More.50k
# Less.50k     4847       97
# More.50k      100     1468


############################
# smote sample with roc

control5$sampling = "smote"
smote_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                           verbose = F, metric = "ROC", 
                  trControl = control5)

pred = predict(smote_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


# Prediction Less.50k More.50k
# Less.50k     4770      174
# More.50k      163     1405


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

# Prediction Less.50k More.50k
# Less.50k     4882       62
# More.50k      133     1435


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


# Prediction Less.50k More.50k
# Less.50k     3993      951
# More.50k      230     1338

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
confusionMatrix(ytest_origin, pred, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4355      589
# More.50k       87     1481



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
# Less.50k     4810      134
# More.50k      278     1290




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
# Less.50k     4792      152
# More.50k      249     1319



model_list4 = list(weighted = weighted_fit,
                   weighted_strata = weighted_strata,
                   down_weight = weighted_down_fit,
                   down_wight_wt_rp = weighted_down_fit2,
                   SMOTE_weight = weighted_smote_fit)


custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
model_roc_plot(model_list4, custom_col)





############################################
# tuning parameter:


rf_random = train(xtrain_origin, ytrain_origin, method = "rf", 
                  verbose = F, metric = "ROC", tuneLength = 2:10,
                  trControl = control5)
###rf_random results:

# Prediction Less.50k More.50k
# Less.50k     3895     1049
# More.50k      230     1338


pred = predict(rf_random, test_final[,-c("income")])
confusionMatrix(test_final$income, pred, positive = "More.50k")



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

pred = predict(rf_gridsearch, test_final[,-c("income")])
confusionMatrix(test_final$income, pred, positive = "More.50k")







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


