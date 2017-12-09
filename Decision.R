# Build a Classification Tree
# 
# Fit a classification tree (see examples in ISL chapter 8, and APM chapter 14).
# Make plots and describe the steps you took to justify choosing optimal tuning parameters.
# Report your 5 (or 6 or 7) important features (could be either just 5, or 6 or 7), with their variable importance statistics.
# Report the training accuracy rate.
# Plot the ROC curve, and report its area under the curve (AUC) statistic.

############################################################# Setting up the data
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
library(data.table)
set.seed(123)
df_impute = readRDS("data/df_impute_feat.rds")
df_impute = data.table(df_impute)
#saveRDS(df_impute,"data/df_impute_feat.rds")
train_idx = createDataPartition(df_impute$income, p =.8)[[1]]
train_origin = df_impute[train_idx,]
test_origin = df_impute[-train_idx, ]
xtrain_origin  = train_origin[,-c("income")]
ytrain_origin = train_origin$income
xtest_origin  = test_origin[,-c("income")]
ytest_origin = test_origin$income

###########################

seed = 1234
set.seed(seed)
library(tree)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
train_origin <- data.frame(train_origin)
test_origin <- data.frame(test_origin)
tree.inc <- tree(formula = income~., data = train_origin)
summary(tree.inc)

# Classification tree:
#   tree(formula = income ~ ., data = train_origin)
# Variables actually used in tree construction:
#   [1] "relationship" "capital.gain" "Edu_Mean_inc"
# [4] "occupation"  
# Number of terminal nodes:  8 
# Residual mean deviance:  0.7056 = 18370 / 26040 
# Misclassification error rate: 0.1555 = 4051 / 26049 

#Training error rate is 15.55%

tree.pred <- predict(tree.inc, test_origin, type = 'class')
table(tree.pred, ytest_origin)
# tree.pred  Less.50k More.50k
# Less.50k     4659      815
# More.50k      256      782
(4659 + 782) / sum(table(tree.pred, ytest_origin)) #correct prediction rate is 83.55%
confusionMatrix(ytest_origin, tree.pred, positive = "More.50k")

cv.inc <- cv.tree(tree.inc, FUN = prune.misclass)
names(cv.inc)
cv.inc

#Lowest cv error rate is at 8 or 5 nodes, with 4040 cv errors. 
#Plotting size and k by dev

par(mfrow = c(1,2))
plot(cv.inc$size, cv.inc$dev, type = 'b')
plot(cv.inc$k, cv.inc$dev, type = 'b')
#We see an elbow at 5 for size

#Pruning the tree by size:
prune.inc <- prune.misclass(tree.inc, best = 5, newdata = test_origin)
plot(prune.inc)
text(prune.inc, pretty = 0)

tree.pred <- predict(prune.inc, test_origin, type = 'class')
table(tree.pred, ytest_origin)
# tree.pred  Less.50k More.50k
# Less.50k     4661      866
# More.50k      254      731
(4661 +731)/sum(table(tree.pred, ytest_origin)) #correct prediction rate is now 82.8%
confusionMatrix(ytest_origin, tree.pred, positive = "More.50k")

#Prune the tree by size and k:
k = 218
prune.inc_1 <- prune.misclass(tree.inc, best = 5, newdata = test_origin, k = 218)
plot(prune.inc_1)
text(prune.inc_1, pretty = 0)

tree.pred_1 <- predict(prune.inc_1, test_origin, type = 'class')
table(tree.pred_1, ytest_origin)
# tree.pred  Less.50k More.50k
# Less.50k     4661      866
# More.50k      254      731
(4661 +731)/sum(table(tree.pred_1, ytest_origin)) #correct prediction rate is now 82.8%

#k = 365
prune.inc_2 <- prune.misclass(tree.inc, best = 5, newdata = test_origin, k = 365)
plot(prune.inc_2)
text(prune.inc_2, pretty = 0)

tree.pred_2 <- predict(prune.inc_2, test_origin, type = 'class')
table(tree.pred_2, ytest_origin)
# tree.pred  Less.50k More.50k
# Less.50k     4663      973
# More.50k      252      624
(4663 +624)/sum(table(tree.pred_2, ytest_origin)) #correct prediction rate is now 81.18%; lower specificity and sensisitivty

#Original parameters were the best 


##Try via rpart
library(rpart)
cartModel <- rpart(income ~., train_origin)
cart.pred_1 <- predict(cartModel, test_origin, type = 'class')
table(cart.pred_1, ytest_origin)
    # cart.pred_1 Less.50k More.50k
    # Less.50k     4664      770
    # More.50k      252      826
(4659 + 782)/sum(table(cart.pred_1, ytest_origin))
confusionMatrix(ytest_origin, cart.pred_1, positive = "More.50k")

#examining tree a bit more
cartModel$variable.importance

printcp(cartModel)
#As a rule of thumb, it’s best to prune a decision tree using the cp of smallest tree that is within one standard deviation of the tree with the smallest xerror.
#Best xerror is 0.64958 with xstd of 0.009372, so we want the smallest tree with an error less than 0.658952
#This is the 4th tree, with cp = 0.010
#Test error is: 0.006064632

min.xerror <- cartModel$cptable[which.min(cartModel$cptable[,"xerror"]),"CP"]

cartPrune <- prune(cartModel, cp = min.xerror) 
cart.pred <- predict(cartPrune, newdata = test_origin, type="class") #Returns the predicted class
cart.pred.prob <- predict(cartPrune, newdata = test_origin, type="prob") #Returns a matrix of predicted probabilities
table(cart.pred, ytest_origin) #exact same as before
      # cart.pred  Less.50k More.50k
      # Less.50k     4664      770
      # More.50k      252      826


##Conclusion: pruning by CP doesn't much with our processed data
#We'll use cartModel as our decision tree

#Accuracy rate is 
(4659 + 782)/sum(table(cart.pred_1, ytest_origin))

#Plot the ROC curve, and report its area under the curve (AUC) statistic.
library(ROCR)

pred_ROC <-prediction(cart.pred_1, ytest_origin)
perf <- performance(pred_ROC, measure = "tpr", x.measure = "fpr")
plot(perf)
abline(coef=c(0,1), col = "grey")
AUC <- performance(pred_ROC,"auc")
AUC@y.values


##Looking at most important variables:
cartModel$variable.importance
plot(cartModel)
text(cartModel, cex = 0.5)
plotcp(cartModel)

ploting <- varImp(cartModel)/sum(varImp(cartModel))
ploting <- t(ploting)
plotting <- ploting[order(ploting, decreasing = TRUE)]
names(plotting) <- c(colnames(ploting)[order(ploting, decreasing = TRUE)])
barplot(plotting, cex.names  = 0.8, names.arg=names(ploting),las=2)

#Top 7 most important variables:
plotting[1:7]


