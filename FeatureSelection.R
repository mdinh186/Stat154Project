library(rpart)
library(randomForest)
library(data.table)
library(caret)

#############################################################
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
set.seed(123)

df_impute = readRDS("data/df_remove_feat.rds")



n = ceiling(nrow(df_impute) * 0.8)
#############################################################
# Prepocessing data:
#############################################################



train_idx = sample(nrow(df_impute),n)
train_origin = df_impute[train_idx,]
test_origin = df_impute[-train_idx, ]
xtrain_origin  = train_origin[,-c("income")]
ytrain_origin = train_origin$income

xtest_origin  = test_origin[,-c("income")]
ytest_origin = test_origin$income


#######################################
#make baseline models and feature sections

control = trainControl(method = "cv", number = 5)

rf_default  = train(x = xtrain_origin, y = ytrain_origin,
                  method = "rf", trControl = control, 
                    importance= T, ntree = 100)




rf_default
pred = predict(rf_default, xtest_origin)
confusionMatrix(ytest_origin, pred)
varImp(rf_default)
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

