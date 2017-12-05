library(rpart)
library(randomForest)
library(data.table)
library(caret)

#############################################################
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
df_origin  = read.table("data/adult.data", sep = ",",colClasses=c("numeric", "character", "numeric", "factor","numeric", "factor", "character", "factor", "factor", "factor", rep("numeric", 3),"character","factor"))
df_origin = data.table(df_origin)
names(df_origin) = c("age", "workclass", "fnlwgt", "education", "education-num", "martial-status", 
              "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hour-per-week", "native_country","income")
set.seed(123)
n = ceiling(nrow(df_origin) * 0.8)
#############################################################
# Prepocessing data:
#############################################################

#convert character vector into factor: 
origin_char = c("workclass", "occupation", "native_country")
for (col in origin_char){
  df[[col]] = as.factor(df[[col]])
}

transform_char = c("workclass", "martial-status", "occupation","race", "sex", "gen_race", "gen_mrg")


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