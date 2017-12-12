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
library(ggplot2)

#############################################################
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
set.seed(123)


#############################################################
cl = makeCluster(detectCores()-1)
registerDoParallel(cl)
#############################################################

library(data.table)
set.seed(123)
train_origin = readRDS("data/train.RDS")
test_origin = readRDS("data/test.rds")

train_origin <- data.table(train_origin)
test_origin <- data.table(test_origin)

xtrain_origin <- train_origin[,-c("income")]
ytrain_origin <- train_origin$income
xtest_origin <- test_origin[,-c("income")]
ytest_origin <- test_origin$income

train_origin <- data.frame(train_origin)
test_origin <- data.frame(test_origin)
xtrain_origin <- data.frame(xtrain_origin)
xtest_origin <- data.frame(xtest_origin)
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
confusionMatrix(pred, ytest_origin, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k     4589      598
# More.50k     355       970
# 
# Accuracy : 0.8537          
# Sensitivity : 0.6186          
# Specificity : 0.9282
# FP/N: 598/(598+970) = 0.3813776


varImp(rf_default)
    # capital.gain       100.000
    # capital.loss        58.478
    # relationship        48.586
    # Edu_Mean_inc        43.327
    # occupation          29.519
    # age                 20.234
    # hours.per.week      16.426
    # gen_race             8.870
    # workclass            7.654
    # marital_status       6.548
    # occ_sex              4.421
    # Gen_Med_Mrg_Inc      3.789
    # native_country       2.195
    # Race_Med_Inc         1.748
    # Gen_Med_Inc          1.510
    # fnlwgt               0.000
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
confusionMatrix(pred, ytest_origin, positive = "More.50k")
#       Prediction Less.50k More.50k
#       Less.50k     4065      235
#       More.50k      879     1333
#       
#       Accuracy : 0.8289          
#       Sensitivity : 0.8501          
#       Specificity : 0.8222          
#       FP/N: 879/(879+4065): 0.1777913

#Over sampling
ctrol2 = trainControl(method =  "cv", number =5, 
                      verboseIter = F,
                      sampling = "up")
model_rf_over = train(xtrain_origin, ytrain_origin, method = "rf",
                      trControl = ctrol2, tuneGrid = tunegrid, ntree=200)
pred = predict(model_rf_over, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")

# Prediction Less.50k More.50k
# Less.50k     4488      488
# More.50k      456     1080
 
# Accuracy : 0.855           
# Sensitivity : 0.7031          
# Specificity : 0.9019          
# FP/N: 456/(4488+456) = 0.09223301

####  
model_list1 = list(original = rf_default,
                   down = model_rf_under, 
                   up = model_rf_over)

custom_col = c("#000000", "#009E73", "#990022")

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
confusionMatrix(pred, ytest_origin, positive = "More.50k")

      # Prediction Less.50k More.50k
      # Less.50k     3976      218
      # More.50k      968     1350
      # 
      # Accuracy : 0.8179
      # Sensitivity : 0.8610          
      # Specificity : 0.8042 
      # FP/N: 0.1390306
############################
# with down sample + ROC
control5$sampling = "down"
down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                 verbose = F, metric = "ROC", tuneGrid = tunegrid, ntree=200, 
                 trControl = control5)
pred = predict(down_fit, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")
      # Prediction Less.50k More.50k
      # Less.50k     4085      264
      # More.50k      859     1304
      # Accuracy : 0.8275          
      # Sensitivity : 0.8316          
      # Specificity : 0.8263          
      # FP/N: 859/(859+4085) = 0.173746

############################
# up sample with ROC
control5$sampling = "up"
up_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
               verbose = F, metric = "ROC",  tuneGrid = tunegrid, ntree= 200,
               trControl = control5)
pred = predict(up_fit, xtest_origin)
confusionMatrix(pred, ytest_origin, positive = "More.50k")
      # Less.50k     4609      586
      # More.50k      335      982
      # 
      # Accuracy : 0.8586          
      # Sensitivity : 0.6263          
      # Specificity : 0.9322
      # FP/N: 335/(335+4609) = 0.0677589

############################
model_list2 = list(original = rf_strata, 
                   down = down_fit, 
                   up = up_fit)
custom_col = c("#000000", "#009E73", "#0072B2")
model_roc_plot(model_list2, custom_col)
model_roc_plot(model_list2, custom_col, AUC = T)

############################
#We select the down_fit with ROC model 
varImp(down_fit)
      # Overall
      # relationship    100.0000
      # fnlwgt           76.5515
      # capital.gain     31.8887
      # Edu_Mean_inc     27.7286
      # hours.per.week   27.3747
      # occupation       25.0592
      # age              19.3836
      # occ_sex          13.9597
      # workclass         8.6607
      # capital.loss      7.1283
      # native_country    4.4077
      # gen_race          4.0314
      # Race_Med_Inc      1.4450
      # marital_status    1.0391
      # Gen_Med_Mrg_Inc   0.3177
      # Gen_Med_Inc       0.0000
png(filename="Image/Bagged_varImp_down_fit.png")
plot(varImp(down_fit), main = "Top features of bagged tree, lower sample with ROC")
dev.off()

#We look at the model with only the top 10 variables
bag_fit_imp = data.frame(feature = rownames(varImp(down_fit)$importance), Score= varImp(down_fit)$importance)
bag_fit_imp = bag_fit_imp[order(bag_fit_imp[,2], decreasing = T),]$feature
top_ten = c(as.character(bag_fit_imp[1:10]), "income")

train_origin <- data.table(train_origin)
df_final = train_origin[,..top_ten]
test_final = data.table(test_origin)[,..top_ten]
bag_fit_feat = train(df_final[,-c("income")], df_final$income, method = "rf", 
                     verbose = F, metric = "ROC", 
                     trControl = control5, tuneGrid = tunegrid)
pred = predict(bag_fit_feat, test_final[,-c("income")])
confusionMatrix(pred, test_final$income, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k     4052      273
# More.50k      892     1295
# Accuracy : 0.8211          
# Sensitivity : 0.8259          
# Specificity : 0.8196 
# FP/N = 892/(892+4052) = 0.1804207



old.par <- par(mar = c(0, 0, 0, 0))
par(old.par)
png(filename="Bagged_final_ROC.png")
plot.roc(test_origin$income, predict(bag_fit_feat, test_final[,-c("income")], type = "prob")[, "More.50k"],
         xlab = "FPR", ylab = "TPR",
         main ="Bagged Tree final model, top 10 features \nof down sample with ROC",
         print.auc = T)
dev.off()
