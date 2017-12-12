############################################################# Setting up the data
setwd("~/Documents/Academics/Stat 154/Stat154Project")
library(data.table)
set.seed(123)
train_origin = readRDS("data/train.RDS")
actual_test = readRDS("data/df_impute_test.rds")
actual_test <- data.table(actual_test)

colnames(actual_test) <- colnames(train_origin)
xactual_test <- actual_test[,-c("income")]
train_origin <- data.frame(train_origin)
actual_test <- data.frame(actual_test)
yactual_test <- actual_test$income
###########################
library(tree)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(rpart)
library(caret)

#Final model was CartModel
cartModel <- rpart(income ~., train_origin)

cart_test.pred <- predict(cartModel, xactual_test, type = 'class')
cart_test.prob <- predict(cartModel, xactual_test, type = 'prob')
confusionMatrix(cart_test.pred, actual_test$income, positive = "More.50k")
      # Prediction Less.50k More.50k
      # Less.50k    11805      1901
      # More.50k      630      1945
      # 
      # Accuracy : 0.8445       
      # Sensitivity : 0.5057          
      # Specificity : 0.9493          
      # FP/N is 0.05066345
###############ROC curve

old.par <- par(mar = c(0, 0, 0, 0))
par(old.par)
png(filename="TEST_Decision_ROC.png")
plot.roc(yactual_test, cart_test.prob[, "More.50k"],
         xlab = "FPR", ylab = "TPR",
         main ="Decision Tree ROC on Against Test Set",
         print.auc = T)
dev.off()

