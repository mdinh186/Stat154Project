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
#model with top ten features
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
set.seed(123)

top_ten = readRDS("data/topten2.rds")
top_ten = top_ten[1:10]

train_origin = readRDS("data/train.RDS")
test_origin = readRDS("data/test.rds")
xtrain_origin = train_origin[,..top_ten]
ytrain_origin = train_origin$income

xtest_origin = test_origin[,..top_ten]
ytest_origin = test_origin$income


df_origin  = readRDS("data/df_remove.rds")
train_idx = createDataPartition(df_origin$income, p = 0.8)[[1]]
xtrain_origin =df_origin[train_idx, -c("income")]
ytrain_origin =df_origin[train_idx,]$income
xtest_origin=df_origin[-train_idx, -c("income")]
ytest_origin = df_origin[-train_idx,]$income




control = trainControl(method = "cv", number = 5)
rf_default  = train(x = xtrain_origin, y = ytrain_origin,
                    method = "rf", trControl = control, 
                    importance= T, ntree = 100)

pred = predict(rf_default, xtest_origin)
confusionMatrix(pred, ytest_origin,  positive = "More.50k")


##########################################################################################
#Sampling technique for imbalanced class: 
##########################################################################################
######Under sampling: 


ctrol = trainControl(method = "cv", number =5, 
                     verboseIter = F,
                     sampling = "down")
model_rf_under = train(xtrain_origin, ytrain_origin, method = "rf",
                       trControl = ctrol)



##### Over sampling: 
ctrol2 = trainControl(method = "cv", number =5, 
                      verboseIter = F,
                      sampling = "up")
model_rf_over = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol2)



#### Smote: 
ctrol4 = trainControl(method = "cv", number =5, 
                      verboseIter = F,
                      sampling = "smote")

model_rf_smote = train(xtrain_origin, ytrain_origin, method = "rf",
                       trControl = ctrol4)



model_list_samp = list(base_line = rf_default,
                      under_sampling = model_rf_under,
                       over_sampling = model_rf_over,
                       smote_sampling = model_rf_smote)
custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00")
model_roc_plot(model_list_samp, custom_col, AUC = T)

#############################################################
# Use ROC metric
#############################################################

# stratified sample with ROC 

control5 = trainControl(method = "cv", number = 5, search = "random",allowParallel = TRUE, 
                        summaryFunction = twoClassSummary, 
                        classProbs = T)

rf_strata = train(xtrain_origin, ytrain_origin, method = "rf", 
                  tuneLength = 15, trControl=control5, 
                  strata = ytrain_origin, sampsize = c(50,50), 
                  metric = "ROC")



# with down sample + ROC

control5$sampling = "down"
down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                 verbose = F, metric = "ROC", 
                 trControl = control5)



# up sample with roc
control5$sampling = "up"
up_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
               verbose = F, metric = "ROC", 
               trControl = control5)

# smote sample with roc

control5$sampling = "smote"
smote_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                  verbose = F, metric = "ROC", 
                  trControl = control5)



model_list_roc_samp = list(baseline = rf_default,
                           under_sampling_w_roc = down_fit, 
                           over_sampling_w_roc= up_fit, 
                           smote_sampling_w_roc = smote_fit, 
                           stratified_50_sampling_w_roc = rf_strata)
custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
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


# with weight  and strata: 
weighted_strata = train(xtrain_origin, ytrain_origin, method = "rf", 
                        tuneLength = 13, trControl=ctrol8, 
                        strata = ytrain_origin, 
                        sampsize = c(50,50), 
                        metric = "ROC",
                        weights = model_weights)



ctrol8 = trainControl(method = "cv", number =5, 
                      summaryFunction = twoClassSummary, 
                      classProbs = T)
nmin = min(table(ytrain_origin))
weighted_down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                          tuneLength = 15, trControl=ctrol8, 
                          strata = ytrain_origin, 
                          sampsize = rep(nmin, 2), metric = "ROC",
                          weights = model_weights)


nmin = min(table(ytrain_origin))
weighted_down_fit2 = train(xtrain_origin, ytrain_origin, method = "rf", 
                           tuneLength = 15, trControl=ctrol8, 
                           strata = ytrain_origin, 
                           metric = "ROC",
                           replace = F,
                           weights = model_weights)


ctrol8$sampling = "smote"
weighted_smote_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                           tuneLength = 15, 
                           trControl=ctrol8, 
                           strata = ytrain_origin, 
                           metric = "ROC",
                           weights = model_weights)

model_list_weight_samp = list(baseline = rf_default,
                           weighted = weighted_fit, 
                           weighted_down_samp_w_roc= weighted_down_fit, 
                           weighted_smote_sampling_w_roc = weighted_smote_fit, 
                           weighted_stratified_sampling_w_roc = weighted_down_fit2)
custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
model_roc_plot(model_list_weight_samp, custom_col, AUC = T)



##################################################################################
#Tunning the parameter with best strategy: 
##################################################################################

####### Tuning mtries and trees: 
##### random search: 
control5$search = "random"

rf_random = train(xtrain_origin, ytrain_origin, method = "rf", 
                  verbose = F, metric = "ROC", tuneLength = 2:9,
                  trControl = control5)



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

model_list_tuning = list(baseline = rf_default,
                              random_search_smote= rf_random, 
                              grid_search_smote = rf_gridsearch)
custom_col = c("#000000", "#009E73", "#0072B2")
model_roc_plot(model_list_tuning, custom_col, AUC = T)



