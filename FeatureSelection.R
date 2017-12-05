library(rpart)
library(randomForest)
library(data.table)
library(caret)

#############################################################
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
set.seed(123)

df_impute = readRDS("data/df_impute_feat.rds")
df_impute$marital_status = as.factor(df_impute$marital_status)
df_impute$sex = as.factor(df_impute$sex)



n = ceiling(nrow(df_impute) * 0.8)
#############################################################
# Prepocessing data:
#############################################################



train_idx = sample(nrow(df_impute),n)
train_origin = df_impute[train_idx,]
test_origin = df_impute[-train_idx, ]
xtrain_origin  = train_origin[,-c("income")]
ytrain_origin = train_origin$income




#######################################
#make baseline models and feature sections

control = trainControl(method = "cv", number = 5)

rf_default  = train(x = train_origin[,-14], y = train_origin$income,
                  method = "rf", trControl = control, 
                    importance= T, ntree = 100)


### evaluation: 