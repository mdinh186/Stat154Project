library(rpart)
library(randomForest)
library(data.table)
library(caret)
library(doParallel)
library(plotROC)
library(dplyr)
library(purrr)
library(pROC)


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

n = ceiling(nrow(df_impute) * 0.8)
#############################################################
# Prepocessing data:
#############################################################

train_idx = sample(nrow(df_impute),n)
train_origin = df_impute[train_idx,]
test_origin = df_impute[-train_idx, ]
xtrain_origin  = train_origin[,-c("income")]
ytrain_origin = train_origin$income

xtest_origin  = test_origin[,-c("income")]
ytest_origin = test_origin$income


#######################################
#make baseline models and feature sections


control = trainControl(method = "cv", number = 5)
rf_default  = train(x = xtrain_origin, y = ytrain_origin,
                    method = "parRF", trControl = control, 
                    importance= T, ntree = 100, replace= F, mtry = ncol(xtrain_origin))
rf_default
pred = predict(rf_default, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
varImp(rf_default)


########################################################


pred = predict(rf_random, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


##### grid search: 
control2 = trainControl(method = "cv", number = 5, search = "grid", allowParallel = TRUE)
tunegrid = expand.grid(.ntress = seq(50, 300, 50))
rf_gridsearch = train(xtrain_origin, ytrain_origin, method = "parRF",
                      tuneGrid=tunegrid, mtry = ncol(xtrain_origin),
                      trControl=control2)

###rf_gridsearch results

# The final value used for the model was mtry = 4.

pred_grid = predict(rf_random, xtest_origin)
confusionMatrix(ytest_origin, pred_grid, positive = "More.50k")



#############################################
#Sampling technique for imbalanced class: 
#############################################
######Under sampling: 


ctrol = trainControl(method = "cv", number =5, 
                     verboseIter = F,
                     sampling = "down")
set.seed(123)
model_rf_under = train(xtrain_origin, ytrain_origin, method = "parRF",
                       trControl = ctrol, mtry = ncol(xtrain_origin))

pred = predict(model_rf_under, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


ctrol2 = trainControl(method = "cv", number =5, 
                      verboseIter = F,
                      sampling = "up")

model_rf_over = train(xtrain_origin, ytrain_origin, method = "parRF",
                      trControl = ctrol2, mtry = ncol(xtrain_origin))


pred = predict(model_rf_over, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


# Reference
# Prediction Less.50k More.50k
# Less.50k     3928      993
# More.50k      270     1321


#### Smote: 
ctrol4 = trainControl(method = "cv", number =5, 
                      verboseIter = F,
                      sampling = "smote")

model_rf_smote = train(xtrain_origin, ytrain_origin, method = "parRF",
                       trControl = ctrol4, mtry = ncol(xtrain_origin))

pred = predict(model_rf_smote, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")






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

rf_strata = train(xtrain_origin, ytrain_origin, method = "parRF", 
                  mtry = ncol(xtrain_origin), trControl=control5, 
                  strata = ytrain_origin, sampsize = c(50,50), 
                  metric = "ROC")


pred = predict(rf_strata, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


# with down sample + ROC

control5$sampling = "down"
down_fit = train(xtrain_origin, ytrain_origin, method = "parRF", 
                 verbose = F, metric = "ROC", mtry = ncol(xtrain_origin), 
                 trControl = control5)

pred = predict(down_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


############################
# up sample with roc
control5$sampling = "up"
up_fit = train(xtrain_origin, ytrain_origin, method = "parRF", 
               verbose = F, metric = "ROC",  mtry = ncol(xtrain_origin),
               trControl = control5)

pred = predict(up_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")



############################
# smote sample with roc

ctrol8$sampling = "smote"
smote_fit = train(xtrain_origin, ytrain_origin, method = "parRF", 
                  verbose = F, metric = "ROC",  mtry = ncol(xtrain_origin),
                  trControl = ctrol8)

pred = predict(smote_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")



model_list2 = list(original = rf_default,
                   strata = rf_strata,
                   down_roc = down_fit,
                   up_weight = up_fit,
                   SMOTE_weight = smote_fit)


custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
model_roc_plot(model_list2, custom_col)

