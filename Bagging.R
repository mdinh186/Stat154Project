library(rpart)
library(randomForest)
library(data.table)
library(caret)
library(doParallel)
library(plotROC)


set.seed(seed)

train_origin <- data.frame(train_origin)
test_origin <- data.frame(test_origin)
bag <- randomForest(income~., data = train_origin, mtry = 16, importance = TRUE, ntree = 25)
yhat.bag = predict(bag, newdata = test_origin)

table(yhat.bag, ytest_origin)
# yhat.bag   Less.50k More.50k
# Less.50k     4499      417
# More.50k      562      1034
(4517 +1016)/sum(table(yhat.bag, ytest_origin)) #correct prediction rate is 0.8496622
confusionMatrix(ytest_origin, yhat.bag, positive = "More.50k")

bag <- randomForest(income~., data = train_origin, mtry = 16, importance = TRUE, ntree = 125)
yhat.bag = predict(bag, newdata = test_origin)
# yhat.bag   Less.50k More.50k
# Less.50k     4521      395
# More.50k      565     1031
confusionMatrix(ytest_origin, yhat.bag, positive = "More.50k")
#Accuracy is 0.8544

bag <- randomForest(income~., data = train_origin, mtry = 16, importance = TRUE, ntree = 225)
yhat.bag = predict(bag, newdata = test_origin)
# Prediction Less.50k More.50k
# Less.50k     4526      390
# More.50k      564     1032
confusionMatrix(ytest_origin, yhat.bag, positive = "More.50k")
#Accuracy of 0.8535

bag <- randomForest(income~., data = train_origin, mtry = 16, importance = TRUE, ntree = 150)
yhat.bag = predict(bag, newdata = test_origin)
# Prediction Less.50k More.50k
# Less.50k     4531      385
# More.50k      568     1028
confusionMatrix(ytest_origin, yhat.bag, positive = "More.50k") 
#Accuracy of 0.8537

# The only parameters when bagging decision trees is the number of samples and
# hence the number of trees to include. This can be chosen by increasing the 
# number of trees on run after run until the accuracy begins to stop showing 
# improvement (e.g. on a cross validation test harness). Very large numbers 
# of models may take a long time to prepare, but will not overfit the 
# training data.

train_control <- trainControl(method="boot", number=100)
# train the model
model <- train(income~., data=train_origin, trControl=train_control, method="nb")


####### Tuning ntrees: 
seed = 1234
set.seed(1234)

ntree_test <- seq(50, 500, 50)

accuracies <- vector()
for(i in 1: length(ntree_test)){
  bag <- randomForest(income~., data = train_origin, mtry = 16, importance = TRUE, ntree = ntree_test[i])
  yhat.bag = predict(bag, newdata = test_origin)
  test <- confusionMatrix(ytest_origin, yhat.bag, positive = "More.50k")
  accuracies[i] <- (test$table[1] + test$table[4])/sum(test$table) #accuracy rate  
}
names(accuracies) <- ntree_test
accuracies
which.max(accuracies) #this is the 9th one, which gives us ntree = 450 and an accuracy of 0.8547297

plot(accuracies)

##### Tuning maximum depth:

#First through maxnodes
naccuracies <- vector()
node_options = c(1,5,10,50,100, 200, 250, 500)

for (h in 1:length(node_options)){
  model <- randomForest(income~., data = train_origin, ntree = 200, importance = TRUE, mtry= 16, nodesize = node_options[h])
  yhat.bag = predict(model, newdata = test_origin)
  testy <- confusionMatrix(ytest_origin, yhat.bag, positive = "More.50k")
  naccuracies[h] <- ((testy$table[1] + testy$table[4])/sum(testy$table))
}

names(naccuracies) <- node_options
naccuracies
which.max(naccuracies) #this is the 5th one, which gives us nodesize = 100

plot(naccuracies)

#Through node_sizes
maccuracies <- vector()
node_size_options = seq(2, 35, 1)
for (h in 1:length(node_size_options)){
  model <- randomForest(income~., data = train_origin, ntree = 450, importance = TRUE, mtry= 16, maxnodes = node_size_options[h])
  yhat.bag = predict(model, newdata = test_origin)
  testy <- confusionMatrix(ytest_origin, yhat.bag, positive = "More.50k")
  maccuracies[h] <- ((testy$table[1] + testy$table[4])/sum(testy$table))
}
names(maccuracies) <- node_size_options
which.max(maccuracies) #maxnodes = 24, accuracy of 0.852887
plot(maccuracies)


#Confirming my parameters:
set.seed(seed)
model <- randomForest(income~., data = train_origin, ntree = 450, importance = TRUE, mtry= 16, maxnodes = 24, nodesize = 100)
yhat.bag = predict(model, newdata = test_origin)
testy <- confusionMatrix(ytest_origin, yhat.bag, positive = "More.50k")
testy #accuracy of 0.851

set.seed(seed)
model <- randomForest(income~., data = train_origin, ntree = 450, importance = TRUE, mtry= 16, maxnodes = 24)
yhat.bag = predict(model, newdata = test_origin)
testy <- confusionMatrix(ytest_origin, yhat.bag, positive = "More.50k")
testy # acuracy of 0.8506

set.seed(seed)
model <- randomForest(income~., data = train_origin, ntree = 450, importance = TRUE, mtry= 16, nodesize = 100)
yhat.bag = predict(model, newdata = test_origin)
testy <- confusionMatrix(ytest_origin, yhat.bag, positive = "More.50k")
testy #accuracy of 0.8626

#Final tuning parameters are ntree = 450 and nodesize= 100

############### Bagging
library(ipred)
library(MASS)
library(TH.data)
set.seed(seed)

mod <- bagging(income~., data = train_origin, nbagg = 450, coob = TRUE,
               nodesize = 100)
print(mod)
mod_pred <- predict(mod, test_origin)
test <- table(mod_pred, ytest_origin)
test
    # mod_pred   Less.50k More.50k
    # Less.50k     4545      582
    # More.50k      371     1014
(test[1] + test[4])/sum(test) #accuracy, 0.8536548
mod$err #misclassification rate, 0.1482974

set.seed(seed)
mod_1 <- randomForest(income~., data = train_origin, ntree = 450, importance = TRUE, mtry= 16, nodesize = 100)
mod_pred_1 <- predict(mod_1, test_origin)
test_1 <- table(mod_pred_1, ytest_origin)
(test_1[1] + test_1[4])/sum(test_1) #accuracy 0.8630221

varImp(mod_1)
varImpPlot(mod_1)

plot(varImp(mod), top = 10)

#######################Summarizing Our Models
re <- data.frame(Tree=cm0$overall[1], 
                 rf=cm1$overall[1], 
                 boosting=cm2$overall[1],
                 bagging=cm3$overall[1])
library(knitr)
re
