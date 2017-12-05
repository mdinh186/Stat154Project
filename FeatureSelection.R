library(rpart)
library(randomForest)
library(data.table)
library(caret)

#############################################################
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
set.seed(123)

df_origin = readRDS("data/df_remove_feat.rds")

n = ceiling(nrow(df_origin) * 0.8)
#############################################################
# Prepocessing data:
#############################################################


#############################################################

train_idx = sample(nrow(df_origin),n)
train_origin = df_origin[train_idx,]
test_origin = df_origin[-train_idx, ]
xtrain_origin  = train_origin[,-c("income")]
ytrain_origin = train_origin$income


train_trans = df_feat[train_idx,]
test_trans = df_feat[-train_idx,]







#######################################
#make baseline models and feature sections

control = trainControl(method = "cv", number = 5)

rf_default  = train(x = xtrain_origin, y = ytrain_origin,
                    metric = "logLoss", method = "rf", trControl = control, 
                    importance= T)