library(rpart)
library(randomForest)
library(data.table)
library(caret)
library(doParallel)
library(plotROC)


#############################################################
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
set.seed(123)

df_impute = readRDS("data/df_impute_feat.rds")
df_impute = data.table(df_impute)

#saveRDS(df_impute,"data/df_impute_feat.rds")

n = ceiling(nrow(df_impute) * 0.8)
#############################################################
# Prepocessing data:
#############################################################
col = c( "workclass", "marital_status", "occupation", "relationship", "native_country","gen_race")
df_impute[,(col):= lapply(.SD, function(x) as.factor(x)), .SDcols =col]
df_impute$sex = NULL
train_idx = sample(nrow(df_impute),n)
train_origin = df_impute[train_idx,]
test_origin = df_impute[-train_idx, ]
xtrain_origin  = train_origin[,-c("income")]
ytrain_origin = train_origin$income

xtest_origin  = test_origin[,-c("income")]
ytest_origin = test_origin$income


#######################################
#make baseline models and feature sections



control = trainControl(method = "cv", number = 5,savePredictions = T,classProbs=T,summaryFunction=twoClassSummary)

rf_default  = train(x = xtrain_origin, y = ytrain_origin,
                  method = "rf", trControl = control, 
                    importance= T, ntree = 100)



###
selectedIndices = rf_default$pred$mtry == 9

g <- ggplot(rf_default$pred[selectedIndices, ], aes(m=More.50k, d=factor(obs, levels = c("Less.50k","More.50k")))) + 
  geom_roc(n.cuts=0) + 
  coord_equal() +
  style_roc()

g + annotate("text", x=0.75, y=0.25, label=paste("AUC =", round((calc_auc(g))$AUC, 4)))


rf_default
pred = predict(rf_default, xtest_origin, cutoff = 0.3)
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
#### Accuary: train:82.7?%

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
seed = 1234
set.seed(1234)

control = trainControl(method = "cv", number = 5, search = "random",allowParallel = TRUE)
mtry = sqrt(ncol(xtrain_origin))
rf_random = train(xtrain_origin, ytrain_origin, method = "rf", 
                  tuneLength = 15, trControl=control)


pred = predict(rf_random, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
###################################
# Prediction Less.50k More.50k
# Less.50k     4602      346
# More.50k      560     1004


##### grid search: 
control2 = trainControl(method = "cv", number = 5, search = "grid", allowParallel = TRUE)
tunegrid = expand.grid(.mtry = c(4:7))
rf_gridsearch = train(xtrain_origin, ytrain_origin, method = "rf",
                      tuneGrid=tunegrid, trControl=control2)


