############################################################# Setting up the data
setwd("~/Documents/Academics/Stat 154/Stat154Project")
library(data.table)
set.seed(123)
train_origin = readRDS("data/train.RDS")
actual_test = readRDS("data/df_impute_test.rds")

train_origin <- data.frame(train_origin)

xtrain_origin <- train_origin[,-c("income")]
ytrain_origin <- train_origin$income
xactual_test <- actual_test[,-c("income")]
yactual_test <- actual_test$income

train_origin <- data.frame(train_origin)
test_origin <- data.frame(test_origin)
xtrain_origin <- data.frame(xtrain_origin)
xtest_origin <- data.frame(xtest_origin)

###################################### Libraries
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

############################
# Final model, with down sample + ROC
set.seed(123)
control5 = trainControl(method = "cv", number = 5, search = "random",allowParallel = TRUE, 
                        summaryFunction = twoClassSummary, 
                        classProbs = T)
control5$sampling = "down"
down_fit = train(xtrain_origin, ytrain_origin, method = "rf", 
                 verbose = F, metric = "ROC", tuneGrid = tunegrid, ntree=200, 
                 trControl = control5)


bag_fit_imp = data.frame(feature = rownames(varImp(down_fit)$importance), Score= varImp(down_fit)$importance)
bag_fit_imp = bag_fit_imp[order(bag_fit_imp[,2], decreasing = T),]$feature
top_ten = c(as.character(bag_fit_imp[1:10]), "income")

train_origin <- data.table(train_origin)
df_final = train_origin[,..top_ten]
test_final = data.table(test_origin)[,..top_ten]
bag_fit_feat = train(df_final[,-c("income")], df_final$income, method = "rf", 
                     verbose = F, metric = "ROC", 
                     trControl = control5, tuneGrid = tunegrid)
pred = predict(bag_fit_feat, data.frame(xactual_test))
confusionMatrix(pred, yactual_test, positive = "More.50k")
      # Reference
      # Prediction Less.50k More.50k
      # Less.50k    10011      696
      # More.50k     2424     3150
      # Accuracy : 0.8084          
      # Sensitivity : 0.8190          
      # Specificity : 0.8051 
      # FP/N: 2424/(10011+2424) = 0.1949337


###############ROC curve
pred.prob = predict(bag_fit_feat, data.frame(xactual_test), type = "prob")

old.par <- par(mar = c(0, 0, 0, 0))
par(old.par)
png(filename="Image/TEST_Bagged_ROC.png")
plot.roc(yactual_test, pred.prob[, "More.50k"],
         xlab = "FPR", ylab = "TPR",
         main ="Bagged Tree ROC on Against Test Set",
         print.auc = T)
dev.off()

#AUC is 0.897

