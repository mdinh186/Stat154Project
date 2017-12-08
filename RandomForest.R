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
                  method = "rf", trControl = control, 
                    importance= T, ntree = 100, replace= F)
rf_default
pred = predict(rf_default, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
varImp(rf_default)


########################################################
### result for version1:  12/5 
####impute with feature: 
#### Accuracy: train: 86 %, test = 86%, t
### evaluation:
# Prediction      Less.than.50k More.than.50k
# Less.than.50k          6011           364
# More.than.50k           792          1264


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

###### parallel processing: 
cluster = makeCluster(detectCores() -1)
registerDoParallel(cluster)


####### Tuning mtries: 
##### random search: 


control = trainControl(method = "cv", number = 5, search = "random",allowParallel = TRUE)
mtry = sqrt(ncol(xtrain_origin))
rf_random = train(xtrain_origin, ytrain_origin, method = "rf", 
                  tuneLength = 15, trControl=control)

###rf_random results:

      # The final value used for the model was mtry = 4.
      
    
pred = predict(rf_random, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")

###Confusion Matrix and Statistics
      # Accuracy : 0.859           

      # Sensitivity : 0.7424          
      # Specificity : 0.8897          
      # Pos Pred Value : 0.6387          
      # Neg Pred Value : 0.9293          
      # Prevalence : 0.2081          
      # Detection Rate : 0.1545          

################################### This is for the current data set?
# Prediction Less.50k More.50k
# Less.50k     4588      349
# More.50k      569     1006

################################### I think this was the the last dataset (before we input everything?)
# Prediction Less.50k More.50k
# Less.50k     4602      346
# More.50k      560     1004


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
      # Confusion Matrix and Statistics
      # Accuracy : 0.861           
      # 95% CI : (0.8524, 0.8693)
      # No Information Rate : 0.7936          
      # P-Value [Acc > NIR] : < 2.2e-16       
      # 
      # Kappa : 0.6011          
      # Mcnemar's Test P-Value : 2.082e-14       
      # 
      # Sensitivity : 0.7493          
      # Specificity : 0.8901          
      # Pos Pred Value : 0.6394          
      # Neg Pred Value : 0.9317          
      # Prevalence : 0.2064          
      # Detection Rate : 0.1546          




#############################################
#Sampling technique for imbalanced class: 
#############################################
######Under sampling: 


ctrol = trainControl(method = "cv", number =5, 
                     verboseIter = F,
                     sampling = "down")
set.seed(123)
model_rf_under = train(xtrain_origin, ytrain_origin, method = "rf",
                       trControl = ctrol)

pred = predict(model_rf_under, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")



# Confusion Matrix and Statistics
# 
# Reference
# Prediction Less.50k More.50k
# Less.50k     3935      986
# More.50k      257     1334



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
# Less.50k     3928      993
# More.50k      270     1321


#### Smote: 
ctrol4 = trainControl(method = "cv", number =5, 
                               verboseIter = F,
                               sampling = "smote")

model_rf_smote = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol4)

pred = predict(model_rf_smote, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


# Confusion Matrix and Statistics
# 
# Reference
# Prediction Less.50k More.50k
# Less.50k     4657      264
# More.50k      674      917




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
# Less.50k     3865     1056
# More.50k      252     1339

# with down sample + ROC

control5$sampling = "down"
down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                          verbose = F, metric = "ROC", 
                 trControl = control5)

pred = predict(down_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")

# Reference
# Prediction Less.50k More.50k
# Less.50k     3809     1112
# More.50k      196     1395


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
# Less.50k     3915     1006
# More.50k      223     1368

############################
# smote sample with roc

ctrol8$sampling = "smote"
smote_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                           verbose = F, metric = "ROC", 
                  trControl = ctrol8)

pred = predict(smote_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")

##
# Prediction Less.50k More.50k
# Less.50k     4465      456
# More.50k      552     1039


model_list2 = list(original = rf_default,
                   strata = rf_strata,
                   down_roc = down_fit,
                   up_weight = up_fit,
                   SMOTE_weight = smote_fit)


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
# Less.50k     4539      382
# More.50k      593      998




##############################
# with weight  and strata: 

weighted_strata = train(xtrain_origin, ytrain_origin, method = "rf", 
      tuneLength = 15, trControl=ctrol8, 
      strata = ytrain_origin, 
      sampsize = c(50,50), 
      metric = "ROC",
      weights = model_weights)

pred = predict(weighted_strata, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


# Reference
# Prediction Less.50k More.50k
# Less.50k     3855     1066
# More.50k      250     1341

nmin = min(table(ytrain_origin))
weighted_down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                        tuneLength = 15, trControl=ctrol8, 
                        strata = ytrain_origin, 
                        sampsize = rep(nmin, 2), metric = "ROC",
                        weights = model_weights)

pred = predict(weighted_down_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")


# Prediction Less.50k More.50k
# Less.50k     4031      890
# More.50k      257     1334




nmin = min(table(ytrain_origin))
weighted_down_fit2 = train(xtrain_origin, ytrain_origin, method = "rf", 
                          tuneLength = 15, trControl=ctrol8, 
                          strata = ytrain_origin, 
                          metric = "ROC",
                          replace = F,
                          weights = model_weights)

pred = predict(weighted_down_fit2, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")




nmax  = max(table(ytrain_origin))

weighted_up_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                          tuneLength = 15, 
                        trControl=ctrol8, 
                        strata = ytrain_origin, 
                        sampsize = c(8000,8000), metric = "ROC",
                        weights = model_weights)

pred = predict(weighted_up_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")



ctrol8$sampling = "smote"
weighted_smote_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                        tuneLength = 15, 
                        trControl=ctrol8, 
                        strata = ytrain_origin, 
                        metric = "ROC",
                        weights = model_weights)

pred = predict(weighted_smote_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")




model_list4 = list(weighted = weighted_fit,
                   weighted_strata = weighted_strata,
                   down_weight = weighted_down_fit,
                   up_weight = weighted_up_fit,
                   SMOTE_weight = weighted_smote_fit)


custom_col = c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
model_roc_plot(model_list4, custom_col)



##################################################################################
#Tunning the parameter with best strategy: 
##################################################################################
###### parallel processing: 
cluster = makeCluster(detectCores() -1)
registerDoParallel(cluster)


####### Tuning mtries and trees: 
##### random search: 


