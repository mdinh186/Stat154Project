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
# Number of terminal nodes:  7 
# Residual mean deviance:   0.7165 = 18660 / 26040 
# Misclassification error rate: 0.1566 = 4078 / 26049

#Training error rate is 15.66%

tree.pred <- predict(tree.inc, test_origin, type = 'class')
table(tree.pred, ytest_origin)
# tree.pred  Less.50k More.50k
# Less.50k     4687      787
# More.50k      257      781
(4687 + 781) / sum(table(tree.pred, ytest_origin)) #correct prediction rate is 0.8396806
confusionMatrix(ytest_origin, tree.pred, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k     4687      257
# More.50k      787      781
# 
# Accuracy : 0.8397          
# 95% CI : (0.8305, 0.8485)
# No Information Rate : 0.8406          
# P-Value [Acc > NIR] : 0.5885          
# 
# Kappa : 0.5043          
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.7524          
#             Specificity : 0.8562          
#          Pos Pred Value : 0.4981          
#          Neg Pred Value : 0.9480          
#              Prevalence : 0.1594          
#          Detection Rate : 0.1199          
#    Detection Prevalence : 0.2408          
#       Balanced Accuracy : 0.8043  
#1 - 0.8396806 = 0.1603194 for our misclassification rate

cv.inc <- cv.tree(tree.inc, FUN = prune.misclass)
names(cv.inc)
cv.inc

#Lowest cv error rate is at 7 or 5 nodes, with 4060 cv errors. 
#Plotting size and k by dev

png(filename="decision_cv.png")
old.par <- par( no.readonly = TRUE )
par( oma = c( 0, 0, 3, 0 ), mfrow = c(1,2))
plot(cv.inc$size, cv.inc$dev, type = 'b', main = )
plot(cv.inc$k, cv.inc$dev, type = 'b')
mtext("Examining size and k by dev for cv.tree", outer = TRUE)
dev.off()

#We see an elbow at 5 for size

#Pruning the tree by size:
prune.inc <- prune.misclass(tree.inc, best = 5, newdata = test_origin)
plot(prune.inc)
text(prune.inc, pretty = 0)
tree.pred <- predict(prune.inc, test_origin, type = 'class')
table(tree.pred, ytest_origin)
# tree.pred  Less.50k More.50k
# Less.50k     4687      787
# More.50k      257      781


(4687 +781)/sum(table(tree.pred, ytest_origin)) #correct prediction rate is 83.97%
confusionMatrix(ytest_origin, tree.pred, positive = "More.50k")
#Sensitivity : 0.7524          
#Specificity : 0.8562 

#Prune the tree by size and k:
k = 202
prune.inc_1 <- prune.misclass(tree.inc, best = 5, newdata = test_origin, k = 202)
plot(prune.inc_1)
text(prune.inc_1, pretty = 0)

varImp(prune.inc)

tree.pred_1 <- predict(prune.inc_1, test_origin, type = 'class')
table(tree.pred_1, ytest_origin)
# tree.pred  Less.50k More.50k
# Less.50k     4689      854
# More.50k      255      714
(4689 +714)/sum(table(tree.pred_1, ytest_origin)) #correct prediction rate is now 82.97%

#k = 385
prune.inc_2 <- prune.misclass(tree.inc, best = 5, newdata = test_origin, k = 385)
plot(prune.inc_2)
text(prune.inc_2, pretty = 0)
tree.pred_2 <- predict(prune.inc_2, test_origin, type = 'class')
table(tree.pred_2, ytest_origin)
# tree.pred  Less.50k More.50k
# Less.50k     4693      943
# More.50k      251      625
(4693 +625)/sum(table(tree.pred_2, ytest_origin)) #correct prediction rate is now 81.17%; lower specificity and sensisitivty

#Original k was the best, and choosing best = 5


##Try via rpart
library(rpart)
cartModel <- rpart(income ~., train_origin)
cart.pred_1 <- predict(cartModel, test_origin, type = 'class')
table(cart.pred_1, ytest_origin)
# cart.pred_1 Less.50k More.50k
# Less.50k     4687      787
# More.50k      257      781
(4687 + 781)/sum(table(cart.pred_1, ytest_origin))
confusionMatrix(ytest_origin, cart.pred_1, positive = "More.50k")
#       Sensitivity : 0.7524          
#       Specificity : 0.8562  
#examining tree a bit more
cartModel$variable.importance

printcp(cartModel)
#As a rule of thumb, it’s best to prune a decision tree using the cp of smallest tree that is within one standard deviation of the tree with the smallest xerror.
#Best xerror is 0.64958 with xstd of 0.009372, so we want the smallest tree with an error less than 0.658952
#This is the 4th tree, with cp = 0.010
#Test error is: 0.006064632

min.xerror <- cartModel$cptable[which.min(cartModel$cptable[,"xerror"]),"CP"]
min.xerror

cartPrune <- prune(cartModel, cp = min.xerror) 
cart.pred <- predict(cartPrune, newdata = test_origin, type="class") #Returns the predicted class
cart.pred.prob <- predict(cartPrune, newdata = test_origin, type="prob") #Returns a matrix of predicted probabilities
table(cart.pred, ytest_origin) #exact same as before
# cart.pred  Less.50k More.50k
# Less.50k     4687      787
# More.50k      257      781

##Conclusion: pruning by CP doesn't much with our processed data
#We'll use cartModel as our decision tree

png(filename="decision_tree.png")
plot(cartPrune)
text(cartPrune)
dev.off()

png(filename = "decision_plotcp.png")
plotcp(cartModel)
dev.off()


#Accuracy rate is 
(4687 + 781)/sum(table(cart.pred_1, ytest_origin))

#Plot the ROC curve, and report its area under the curve (AUC) statistic.
library(ROCR)


pred_ROC <-prediction((cart.pred.prob[,2]), (ytest_origin))
perf <- performance(pred_ROC, measure = "tpr", x.measure = "fpr")
AUC <- performance(pred_ROC,"auc")
AUC@y.values

par(mar=c(4,4,4,4))

png(filename="images/Decision_ROC.png")
plot(perf, main="Decision tree, ROC plot")
abline(coef=c(0,1), col = "grey")
mtext(paste("AUC is", as.character(round(as.numeric(AUC@y.values), 4))))
dev.off()

##Looking at most important variables:
cartModel$variable.importance


plot(cartModel)
text(cartModel, cex = 0.5)

png(filename = "decision_varimp.png")
par(mar = c(7, 5, 2, 2))
ploting <- cartModel$variable.importance/sum(cartModel$variable.importance)
ploting <- t(ploting)
plotting <- ploting[order(ploting, decreasing = TRUE)]
names(plotting) <- c(colnames(ploting)[order(ploting, decreasing = TRUE)])
barplot(plotting, cex.names  = 0.7, 
        names.arg=names(ploting), mgp = c(3, 0.5, 0),
        main = "Variable Importance for Decision tree", las=2, ylim=c(0, 0.25))

dev.off()

#Top 7 most important variables:
plotting[1:7]


