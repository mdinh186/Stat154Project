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


seed =123
set.seed(seed)
df_impute = readRDS("data/df_impute_feat.rds")
df_impute = data.table(df_impute)

split1 = createDataPartition(df_impute$income, p  = 0.8)[[1]]

train_origin = df_impute[split1,]
test_origin = df_impute[-split1, ]
xtrain_origin = train_origin[, -c("income")]
ytrain_origin = train_origin$income
xtest_origin = test_origin[,-c("income")]
ytest_origin = test_origin$income

#saveRDS(df_impute,"data/df_impute_feat.rds")
#############################################################
# Prepocessing data:
#############################################################

#######################################
#make baseline models and feature sections
tunegrid <- expand.grid(.mtry=16)

control = trainControl(method = "cv", number = 5)
rf_default  = train(x = xtrain_origin, y = ytrain_origin,
                    method = "rf", trControl = control, 
                    importance= T, ntree = 100, tuneGrid = tunegrid, replace= F)
rf_default
pred = predict(rf_default, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
      # Less.50k     4586      358
      # More.50k      591      977
      # 
      # Accuracy : 0.8543          
      # 95% CI : (0.8455, 0.8628)
      # No Information Rate : 0.795           
      # P-Value [Acc > NIR] : < 2.2e-16       
      # 
      # Kappa : 0.5801          
      # Mcnemar's Test P-Value : 5.034e-14       
      #                                           
      #             Sensitivity : 0.7318          
      #             Specificity : 0.8858          
      #          Pos Pred Value : 0.6231          
      #          Neg Pred Value : 0.9276          
      #              Prevalence : 0.2050          
      #          Detection Rate : 0.1500          
      #    Detection Prevalence : 0.2408          
      #       Balanced Accuracy : 0.8088  
varImp(rf_default)
      # capital-gain      100.0000
      # capital-loss       54.2883
      # relationship       51.0024
      # Edu_Mean_inc       39.7952
      # occupation         32.3488
      # age                17.7601
      # hours-per-week     15.0480
      # gen_race            9.3028
      # marital_status      7.4554
      # workclass           6.0712
      # occ_sex             4.6765
      # Gen_Med_Mrg_Inc     2.8052
      # native_country      1.0119
      # Gen_Med_Inc         0.7164
      # Race_Med_Inc        0.4876
      # fnlwgt              0.0000
plot(varImp(rf_default))

########################################################
#TUNING
# The only parameters when bagging decision trees is the number of samples and
# hence the number of trees to include. This can be chosen by increasing the 
# number of trees on run after run until the accuracy begins to stop showing 
# improvement (e.g. on a cross validation test harness).

control <- trainControl(method="cv", number=5, search="grid", summaryFunction = twoClassSummary, classProbs = T)
tunegrid <- expand.grid(.mtry=16)
modellist <- list()
for (ntree in seq(50, 300, 50)) {
  set.seed(seed)
  fit <- train(xtrain_origin, ytrain_origin, method="rf", tuneGrid = tunegrid, metric="ROC", trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}

# compare results
results <- resamples(modellist)
summary(results)
dotplot(results)
pred_50 = predict(modellist[['50']], xtest_origin)
pred_100 = predict(modellist[['100']], xtest_origin)
pred_150 = predict(modellist[['150']], xtest_origin)
pred_200 = predict(modellist[['200']], xtest_origin)
pred_250 = predict(modellist[['250']], xtest_origin)
pred_300 = predict(modellist[['300']], xtest_origin)

confusionMatrix(pred_50, ytest_origin) #0.8557
confusionMatrix(pred_100, ytest_origin) #0.8563
confusionMatrix(pred_150, ytest_origin) #0.856
confusionMatrix(pred_200, ytest_origin) #0.8564
confusionMatrix(pred_250, ytest_origin) #0.8564
confusionMatrix(pred_300, ytest_origin) #0.8564

#############################################
#Sampling technique for imbalanced class: 
#############################################
######Under sampling: 
tunegrid <- expand.grid(.mtry=16)

ctrol = trainControl(method = "cv", number =5, 
                     verboseIter = F,
                     sampling = "down")
set.seed(123)
model_rf_under = train(xtrain_origin, ytrain_origin, method = "rf",
                       trControl = ctrol, tuneGrid = tunegrid, ntree=200)
pred = predict(model_rf_under, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
#       Less.50k     4084      860
#       More.50k      235     1333
#       
#       Accuracy : 0.8318          
#       95% CI : (0.8225, 0.8409)
#       No Information Rate : 0.6632          
#       P-Value [Acc > NIR] : < 2.2e-16       
#       
#       Kappa : 0.5952          
#       Mcnemar's Test P-Value : < 2.2e-16       
#       
#       Sensitivity : 0.6078          
#       Specificity : 0.9456          
#       Pos Pred Value : 0.8501          
#       Neg Pred Value : 0.8261          
#       Prevalence : 0.3368          
#       Detection Rate : 0.2047          
#       Detection Prevalence : 0.2408          
#       Balanced Accuracy : 0.7767 

ctrol2 = trainControl(method = "cv", number =5, 
                      verboseIter = F,
                      sampling = "up")

model_rf_over = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol2, tuneGrid = tunegrid, ntree=200)
pred = predict(model_rf_over, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
        # Prediction Less.50k More.50k
        # Less.50k     4472      472
        # More.50k      500     1068
        # 
        # Accuracy : 0.8507          
        # 95% CI : (0.8418, 0.8593)
        # No Information Rate : 0.7635          
        # P-Value [Acc > NIR] : <2e-16          
        # 
        # Kappa : 0.5892          
        # Mcnemar's Test P-Value : 0.3865          
        # 
        # Sensitivity : 0.6935          
        # Specificity : 0.8994          
        # Pos Pred Value : 0.6811          
        # Neg Pred Value : 0.9045          
        # Prevalence : 0.2365          
        # Detection Rate : 0.1640          
        # Detection Prevalence : 0.2408          
        # Balanced Accuracy : 0.7965   

cl = makeCluster(detectCores()-1)
registerDoParallel(cl)

####  
model_list1 = list(original = rf_default, 
                   down = model_rf_under, 
                   up = model_rf_over)

custom_col = c("#000000", "#009E73", "#0072B2")

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
                  tuneGrid = tunegrid, ntree=200, trControl=control5, 
                  strata = ytrain_origin, sampsize = c(50,50), 
                  metric = "ROC")
    
pred = predict(rf_strata, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
#       Prediction Less.50k More.50k
#       Less.50k     4033      911
#       More.50k      244     1324
#       
#       Accuracy : 0.8226          
#       95% CI : (0.8131, 0.8318)
#       No Information Rate : 0.6568          
#       P-Value [Acc > NIR] : < 2.2e-16     

############################
# with down sample + ROC
control5$sampling = "down"
down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                 verbose = F, metric = "ROC", tuneGrid = tunegrid, ntree=200, 
                 trControl = control5)
pred = predict(down_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
#       Prediction Less.50k More.50k
#       Less.50k     4049      895
#       More.50k      251     1317
#       
#       Accuracy : 0.824  

############################
# up sample with ROC
control5$sampling = "up"
up_fit = train(xtrain_origin, ytrain_origin, method = "parRF", 
               verbose = F, metric = "ROC",  tuneGrid = tunegrid, ntree= 200,
               trControl = control5)
pred = predict(up_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
      # Prediction Less.50k More.50k
      # Less.50k     4463      481
      # More.50k      498     1070
      # 
      # Accuracy : 0.8497

############################
model_list2 = list(original = rf_strata, 
                   down = down_fit, 
                   up = up_fit)
custom_col = c("#000000", "#009E73", "#0072B2")
model_roc_plot(model_list2, custom_col)
model_roc_plot(model_list2, custom_col, AUC = T)

plot(varImp(model_rf_under))

#Highest AUC is for model_rf_under
#       relationship    100.000
#       fnlwgt           79.134
#       capital-gain     32.090
#       Edu_Mean_inc     31.201
#       hours-per-week   28.622
#       occupation       24.553
#       age              19.182
#       occ_sex          15.137
#       workclass         8.444
#       capital-loss      7.119
#       Gen_Med_Mrg_Inc   5.045
#       native_country    4.377
#       gen_race          4.276
#       Race_Med_Inc      1.398
#       marital_status    0.945
#       Gen_Med_Inc       0.000
