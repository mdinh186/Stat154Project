############################################################# Setting up the data
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
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


###########################
library(tree)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

tree.inc <- tree(income~., data = train_origin)
summary(tree.inc)
#      tree(formula = income ~ ., data = train_origin)
# Variables actually used in tree construction:
#   [1] "relationship" "capital.gain"
# [3] "Edu_Mean_inc" "occupation"  
# Number of terminal nodes:  8 
# Residual mean deviance:  0.7123 = 18550 / 26040 
# Misclassification error rate: 0.158 = 4117 / 26049 
tree.pred <- predict(tree.inc, test_origin, type = 'class')
table(ytest_origin, tree.pred)
(4708 + 799) / sum(table(tree.pred, ytest_origin)) #correct prediction rate is 0.8456695
confusionMatrix(tree.pred, ytest_origin, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k     4708      769
# More.50k      236      799
#Accuracy : 0.8457          
#Sensitivity : 0.5096          
#Specificity : 0.9523  

cv.inc <- cv.tree(tree.inc, FUN = prune.misclass)
names(cv.inc)
cv.inc

#Lowest cv error rate is at 8 or 5 nodes, with 4120 cv errors. 
#Plotting size and k by dev

png(filename="decision_cv.png")
old.par <- par( o.readonly = TRUE )
par( oma = c( 0, 0, 3, 0 ), mfrow = c(1,2))
plot(cv.inc$size, cv.inc$dev, type = 'b', main = )
plot(cv.inc$k, cv.inc$dev, type = 'b')
mtext("Examining size and k by dev for cv.tree", outer = TRUE)
dev.off()

#We see an elbow at 5 for size

#Pruning the tree by size:
prune.inc <- prune.misclass(tree.inc, best = 5, newdata = test_origin, k=0)
plot(prune.inc)
text(prune.inc, pretty = 0)
tree.pred <- predict(prune.inc, test_origin, type = 'class')
table(tree.pred, ytest_origin)
(4708 +799)/sum(table(tree.pred, ytest_origin)) #correct prediction rate is 0.8456695
confusionMatrix(tree.pred, ytest_origin, positive = "More.50k")
      #Sensitivity : 0.5096          
      #Specificity : 0.9523
      #FP/N : 236/(236 +4708) = 0.04773463


##Try via rpart
library(rpart)
cartModel <- rpart(income ~., train_origin)
cart.pred <- predict(cartModel, test_origin, type = 'class')
cart.pred.prob <- predict(cartModel, test_origin, type = 'prob')
confusionMatrix(cart.pred,ytest_origin, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k     4708      769
#More.50k      236      799
#Accuracy : 0.8457          
# Sensitivity : 0.5096         
# Specificity : 0.9523
#FP/N = 236/(236+4708) = 0.04773463

#examining tree a bit more
cartModel$variable.importance
    # relationship Gen_Med_Mrg_Inc  marital_status    capital.gain 
    # 1861.25609      1831.20301      1831.20301       762.63259 
    # Edu_Mean_inc        gen_race     Gen_Med_Inc             age 
    # 753.37400       625.54580       586.34810       412.44721 
    # occupation  native_country    capital.loss 
    # 241.49598        16.14221        13.38104 

printcp(cartModel)
#As a rule of thumb, it’s best to prune a decision tree using the cp of smallest tree that is within one standard deviation of the tree with the smallest xerror.
#Best xerror is 0.65678 with xstd of 0.0093883, so we want the smallest tree with an error less than 0.6656855
#This is the 4th tree, with cp = 0.010
#Test error is: 0.006159704

min.xerror <- cartModel$cptable[which.min(cartModel$cptable[,"xerror"]),"CP"]
min.xerror

png(filename = "decision_plotcp.png")
plotcp(cartModel)
dev.off()

png(filename="decision_tree.png")
plot(cartModel)
text(cartModel)
dev.off()

#Accuracy rate is 0.8457
#Plot the ROC curve, and report its area under the curve (AUC) statistic.
library(ROCR)

pred_ROC <-prediction((cart.pred.prob[,2]), (ytest_origin))
perf <- performance(pred_ROC, measure = "tpr", x.measure = "fpr")
AUC <- performance(pred_ROC,"auc")
AUC@y.values

#AUC is 0.850288

par(mar=c(4,4,4,4))

png(filename="Image/Decision_ROC.png")
plot(perf, main="Decision tree, ROC plot")
abline(coef=c(0,1), col = "grey")
mtext(paste("AUC is", as.character(round(as.numeric(AUC@y.values), 4))))
dev.off()

##Looking at most important variables:
cartModel$variable.importance

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

#Final model is CartModel
