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
                    importance= T, ntree = 200, tuneGrid = tunegrid, replace= F)
rf_default
pred = predict(rf_default, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
        # Prediction Less.50k More.50k
        # Less.50k     4556      388
        # More.50k      578      990
        # 
        # Accuracy : 0.8517          
        # 95% CI : (0.8428, 0.8602)
        # No Information Rate : 0.7884          
        # P-Value [Acc > NIR] : < 2.2e-16       
        # 
        # Kappa : 0.5768          
        # Mcnemar's Test P-Value : 1.195e-09       
        #                                           
        #             Sensitivity : 0.7184          
        #             Specificity : 0.8874          
        #          Pos Pred Value : 0.6314          
        #          Neg Pred Value : 0.9215          
        #              Prevalence : 0.2116          
        #          Detection Rate : 0.1520          
        #    Detection Prevalence : 0.2408          
        #       Balanced Accuracy : 0.802 
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
#       Less.50k     4065      879
#       More.50k      235     1333
#       
#       Accuracy : 0.8289         
#       95% CI : (0.8196, 0.838)
#       No Information Rate : 0.6603         
#       P-Value [Acc > NIR] : < 2.2e-16      
#       
#       Kappa : 0.5897         
#       Mcnemar's Test P-Value : < 2.2e-16      
#                                                
#                   Sensitivity : 0.6026         
#                   Specificity : 0.9453 

ctrol2 = trainControl(method =  "cv", number =5, 
                      verboseIter = F,
                      sampling = "up")

model_rf_over = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol2, tuneGrid = tunegrid, ntree=200)
pred = predict(model_rf_over, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
#       Prediction Less.50k More.50k
#       Less.50k     4474      470
#       More.50k      513     1055
#       
#       Accuracy : 0.849           
#       95% CI : (0.8401, 0.8577)
#       No Information Rate : 0.7658          
#       P-Value [Acc > NIR] : <2e-16          
#       
#       Kappa : 0.5832          
#       Mcnemar's Test P-Value : 0.1804          
#       
#       Sensitivity : 0.6918          
#       Specificity : 0.8971          
#       Pos Pred Value : 0.6728          
#       Neg Pred Value : 0.9049          
#       Prevalence : 0.2342          
#       Detection Rate : 0.1620          
#       Detection Prevalence : 0.2408          
#       Balanced Accuracy : 0.7945    

cl = makeCluster(detectCores()-1)
registerDoParallel(cl)

####  
model_list1 = list(original = rf_default, 
                   down = model_rf_under, 
                   up = model_rf_over)

custom_col = c("#000000", "#009E73", "#0072B2")

model_roc_plot(model_list1, custom_col)
model_roc_plot(model_list1, custom_col, AUC = T)

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
# Prediction Less.50k More.50k
      # Less.50k     3970      974
      # More.50k      254     1314
      # 
      # Accuracy : 0.8114          
      # 95% CI : (0.8017, 0.8209)
      # No Information Rate : 0.6486          
      # P-Value [Acc > NIR] : < 2.2e-16       
      # 
      # Kappa : 0.5541          
      # Mcnemar's Test P-Value : < 2.2e-16       
      #                                           
      #             Sensitivity : 0.5743          
      #             Specificity : 0.9399  
############################
# up sample with ROC
control5$sampling = "up"
up_fit = train(xtrain_origin, ytrain_origin, method = "parRF", 
               verbose = F, metric = "ROC",  tuneGrid = tunegrid, ntree= 200,
               trControl = control5)
pred = predict(up_fit, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
      # Prediction Less.50k More.50k
      # Less.50k     4425      519
      # More.50k      479     1089
      # 
      # Accuracy : 0.8467          


############################
model_list2 = list(original = rf_strata, 
                   down = down_fit, 
                   up = up_fit)
custom_col = c("#000000", "#009E73", "#0072B2")
model_roc_plot(model_list2, custom_col)
model_roc_plot(model_list2, custom_col, AUC = T)

############################
#We select the up_fit with ROC model 
varImp(up_fit)
      # Overall
      # relationship    100.0000
      # fnlwgt           91.2567
      # capital-gain     36.8435
      # hours-per-week   31.3418
      # Edu_Mean_inc     31.3302
      # occupation       27.8495
      # Gen_Med_Mrg_Inc  21.6058
      # age              20.9336
      # occ_sex          14.9907
      # capital-loss      9.6929
      # workclass         8.8587
      # native_country    5.2784
      # gen_race          4.7063
      # Race_Med_Inc      1.7721
      # marital_status    0.8181
      # Gen_Med_Inc       0.0000

png(filename="images/Bagged_varImp_up_fit.png")
plot(varImp(up_fit), main = "Top features of bagged tree, upper sample with ROC")
dev.off()
# relationship    100.000
# fnlwgt           78.563
# capital-gain     32.034
# Edu_Mean_inc     31.506
# hours-per-week   28.820
# occupation       22.315
# age              19.806
# occ_sex          14.616
# workclass         8.615
# capital-loss      6.985
# native_country    4.463
# gen_race          4.045
# Gen_Med_Mrg_Inc   2.760
# Race_Med_Inc      1.441
# marital_status    1.210
# Gen_Med_Inc       0.000

#We look at the model with only the top 10 variables
bag_fit_imp = data.frame(feature = rownames(varImp(up_fit)$importance), Score= varImp(up_fit)$importance)
bag_fit_imp = bag_fit_imp[order(bag_fit_imp[,2], decreasing = T),]$feature
top_ten = c(as.character(bag_fit_imp[1:10]), "income")

df_final = df_impute[,..top_ten]
train_final_idx = createDataPartition(df_final$income, p = 0.8)[[1]]
train_final = df_final[train_final_idx,]
test_final = df_final[-train_final_idx, ]
bag_fit_feat = train(train_final[,-c("income")], train_final$income, method = "rf", 
                     verbose = F, metric = "ROC", 
                     trControl = control5, tuneGrid = tunegrid)
pred = predict(bag_fit_feat, test_final[,-c("income")])
confusionMatrix(test_final$income, pred, positive = "More.50k")
      # Prediction Less.50k More.50k
      # Less.50k     3987      957
      # More.50k      246     1322
      # 
      # Accuracy : 0.8153          
      # 95% CI : (0.8056, 0.8246)
      # No Information Rate : 0.65            
      # P-Value [Acc > NIR] : < 2.2e-16       
      # 
      # Kappa : 0.5625          
      # Mcnemar's Test P-Value : < 2.2e-16       
      # 
      # Sensitivity : 0.5801          
      # Specificity : 0.9419   

#AUC is 0.747

old.par <- par(mar = c(0, 0, 0, 0))
par(old.par)
png(filename="Bagged_final_up_ROC.png")
plot.roc(test_origin$income, predict(bag_fit_feat, test_final[,-c("income")], type = "prob")[, "More.50k"],
         xlab = "FPR", ylab = "TPR",
         main ="Bagged Tree final model, top 10 features \nof upper sample with ROC",
         print.auc = T)
dev.off()
