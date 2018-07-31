rm(list=ls(all=TRUE))

HAR <- read.csv("train.csv")#Loading Train Data

dim(HAR)
names(HAR)
str(HAR) # 562 valiables are numbers and 1 target variable
str(HAR$Activity) # 563rd valiable is factor, target (label)
plot(HAR$Activity)
######Splitting data in to train and validate#######
rows3 <- seq(1,nrow(HAR),1)
rows3
set.seed(100)
trainrows <- sample(rows, nrow(HAR)*0.7)
train_data <- HAR[trainrows,]
validate_data <- HAR[-trainrows,]


########Building Decision Tree model########
library(C50)
decisionTree <- C5.0(x = train_data[,1:562], y = train_data[,563])
require(mgcv)
save(decisionTree, file = 'models/DecisionTree.rda')
summary(decisionTree)
names(decisionTree)
#library("partykit")
#plot(decisionTree)

HAR_test <- read.csv("test.csv")


pred_DT_train = predict(decisionTree,newdata=train_data, type="class")
DT_tr_table <- table(pred_DT_train, train_data$Activity)
DT_tr_table
accuracy_DT_train <- sum(diag(DT_tr_table))/sum(DT_tr_table)*100
accuracy_DT_train

pred_DT_validate = predict(decisionTree,newdata=validate_data, type="class")
DT_vatable <- table(pred_DT_validate, validate_data$Activity)
DT_vatable
accuracy_DT_validate <- sum(diag(DT_vatable))/sum(DT_vatable)*100
accuracy_DT_validate


pred_DT_test = predict(decisionTree,newdata=HAR_test, type="class")
DT_te_table <- table(pred_DT_test, HAR_test$Activity)
DT_te_table
accuracy_DT_test <- sum(diag(DT_te_table))/sum(DT_te_table)*100
accuracy_DT_test



#####Buildig SVM Model##########
library(e1071)

SVM_model2 = svm(Activity ~. , data=train_data, method = "C-classification", kernel = "linear", cost = 10,gamma = 0.1)
save(SVM_model2, file = 'models/SVM.rda')





pred_SVM2_train = predict(SVM_model2,newdata=train_data, type="class")
trtable <-table(pred_SVM2_train, train_data$Activity)
accuracy_SVM2_train <- sum(diag(trtable))/sum(trtable)*100
accuracy_SVM2_train
pred_SVM2_validate = predict(SVM_model2,newdata=validate_data, type="class")
vatable <- table(pred_SVM2_validate, validate_data$Activity)
accuracy_SVM2_validate <- sum(diag(vatable))/sum(vatable)*100
accuracy_SVM2_validate
pred_SVM2_test = predict(SVM_model2,newdata=HAR_test, type="class")
ttable <- table(pred_SVM2_test, HAR_test$Activity)
accuracy_SVM2_test <- sum(diag(ttable))/sum(ttable)*100
accuracy_SVM2_test

##########Building random forest on the data #######

install.packages("randomForest")
library(randomForest)

HAR_rf <- randomForest(x = train_data[,1:562], y = train_data[,563], ntree=100, do.trace=20, impotance = T)
save(decisionTree, file = 'models/RandomForest.rda')
summary(HAR_rf)

plot(HAR_rf)
print(HAR_rf)
HAR_rf$predicted
HAR_rf$confusion
HAR_rf$importance
dim(HAR_rf$importance)
round(importance(HAR_rf), 2)
Imp_HAR_rf <- data.frame(HAR_rf$importance)
Imp_HAR_rf

pred_rfmodel_train <-predict(HAR_rf,train_data,type="response")
pred_rfmodel_validate <-predict(HAR_rf,validate_data,type="response")
pred_rfmodel_test <-predict(HAR_rf,HAR_test,type="response")


RF_tr_table <- table(pred_rfmodel_train, train_data$Activity)
RF_tr_table

RF_va_table <- table(pred_rfmodel_validate, validate_data$Activity)
RF_va_table

RF_te_table <- table(pred_rfmodel_test, HAR_test$Activity)
RF_te_table

accuracy_RF_train <- sum(diag(RF_tr_table))/sum(RF_tr_table)*100
accuracy_RF_train

accuracy_RF_validate <- sum(diag(RF_va_table))/sum(RF_va_table)*100
accuracy_RF_validate ### Accuracy on validate data 98.50408

accuracy_RF_test <- sum(diag(RF_te_table))/sum(RF_te_table)*100
accuracy_RF_test #### Accuracy on test data 91.65253



#######Tuning RandomForest model for better results ######

TuneRFmodel <- tuneRF(x = train_data[,1:562], y = train_data[,563], ntreeTry=200, stepFactor=2, improve=0.05, trace=TRUE, plot=TRUE, doBest=FALSE)
save(TuneRFmodel, file = 'models/TunedRandomForest.rda')
TuneRFmodel

best.m <- TuneRFmodel[TuneRFmodel[, 2] == min(TuneRFmodel[, 2]), 1]
print(TuneRFmodel)
print(best.m)

######Building new RF model with mtry value ########

TuneRFmodel2 <-randomForest(x = train_data[,1:562], y = train_data[,563], mtry=best.m, importance=TRUE, ntree=200)
save(TuneRFmodel, file = 'models/TunedRandomForest2.rda')

print(TuneRFmodel2)

pred_rfmodel2_train <-predict(TuneRFmodel2,train_data,type="response")
pred_rfmodel2_validate <-predict(TuneRFmodel2,validate_data,type="response")
pred_rfmodel2_test <-predict(TuneRFmodel2,HAR_test,type="response")

tunedRandomForestTrain <- as.vector(pred_rfmodel2_train)
table(tunedRandomForestTrain)
RF2_tr_table <- table(pred_rfmodel2_train, train_data$Activity)
RF2_tr_table

RF2_va_table <- table(pred_rfmodel2_validate, validate_data$Activity)
RF2_va_table

RF2_te_table <- table(pred_rfmodel2_test, HAR_test$Activity)
RF2_te_table

accuracy_RF2_train <- sum(diag(RF2_tr_table))/sum(RF2_tr_table)*100
accuracy_RF2_train

accuracy_RF2_validate <- sum(diag(RF2_va_table))/sum(RF2_va_table)*100
accuracy_RF2_validate #### accuray 97.73345 on validate data

accuracy_RF2_test <- sum(diag(RF2_te_table))/sum(RF2_te_table)*100
accuracy_RF2_test### after tuning the accuracy is 92.26332 which is ~0.1 less than first RF Model


######KNN Classificaiton############
library(class)

train_withoutclass = subset(train_data,select=-c(Activity))
validate_withoutclass = subset(validate_data,select=-c(Activity))
test_withoutclass = subset(HAR_test,select=-c(Activity))

knn_pred_train = knn(train_withoutclass, train_withoutclass, train_data$Activity, k = 5)

knn_pred_validate = knn(train_withoutclass, validate_withoutclass, train_data$Activity, k = 5)

knn_pred_test = knn(train_withoutclass, test_withoutclass, train_data$Activity, k = 5)

plot(knn_pred_test)

a=table(knn_pred_train,train_data$Activity)
a
b=table(knn_pred_validate,validate_data$Activity)
b
c=table(knn_pred_test,HAR_test$Activity)
c
knn_train_accu= sum(diag(a))/nrow(train_withoutclass)*100
knn_validate_accu= sum(diag(b))/nrow(validate_withoutclass)*100
knn_test_accu= sum(diag(c))/nrow(test_withoutclass)*100
knn_train_accu
knn_validate_accu
knn_test_accu ##KNN is giving about 79.94 accuracy on test data
######KNN Classificaiton ############

#######Ensemble - stacking #######
library(data.table)
####1. predicted train data#####
ens_target <- train_data$Activity
str(ens_target)
ens_train_data <- data.table(cbind(knn_pred_train,pred_rfmodel2_train,pred_rfmodel_train,pred_SVM_train,pred_DT_train))
setnames(ens_train_data, old = c('knn_pred_train','pred_rfmodel2_train', 'pred_rfmodel_train','pred_SVM_train', 'pred_DT_train' ), new = c('KNN','RF2', 'RF', 'SVM', 'DT'))
head(ens_train_data)
str(ens_train_data)
ens_train_withtarget <- cbind(ens_train_data,ens_target)
names(ens_train_withtarget)[6] <- c('TARGET')
ens_train_withtarget
head(ens_train_withtarget)
str(ens_train_withtarget)
tail(ens_train_withtarget)
dim(ens_train_withtarget)
####2. predicted validate data#####
ens_target2 <- validate_data$Activity
str(ens_target2)
ens_validate_data <- data.table(cbind(knn_pred_validate,pred_rfmodel2_validate,pred_rfmodel_validate,pred_SVM_validate,pred_DT_validate))
names(ens_validate_data)
setnames(ens_validate_data, old = c('knn_pred_validate','pred_rfmodel2_validate', 'pred_rfmodel_validate','pred_SVM_validate', 'pred_DT_validate' ), new = c('KNN','RF2', 'RF', 'SVM', 'DT'))
head(ens_validate_data)
str(ens_validate_data)
ens_validate_withtarget <- cbind(ens_validate_data,ens_target2)
names(ens_validate_withtarget)[6] <- c('TARGET')
ens_validate_withtarget
head(ens_validate_withtarget)
tail(ens_validate_withtarget)
dim(ens_validate_withtarget)
####3. predicted test data#####
ens_target3 <- HAR_test$Activity
str(ens_target3)
ens_test_data <- data.table(cbind(knn_pred_test,pred_rfmodel2_test,pred_rfmodel_test,pred_SVM_test,pred_DT_test))
setnames(ens_test_data, old = c('knn_pred_test','pred_rfmodel2_test', 'pred_rfmodel_test','pred_SVM_test', 'pred_DT_test' ), new = c('KNN','RF2', 'RF', 'SVM', 'DT'))
head(ens_test_data)
str(ens_test_data)
ens_test_withtarget <- cbind(ens_test_data,ens_target3)

names(ens_test_withtarget)[6] <- c('TARGET')
ens_test_withtarget
head(ens_test_withtarget)
tail(ens_test_withtarget)
dim(ens_test_withtarget)

#######Building SVM  on stacked data#######

ens_svm = svm(TARGET~. , data=ens_train_withtarget, method = "C-classification", kernel = "linear", cost = 10,gamma = 0.1)

dim(ens_svm)

save(ens_svm, file = 'models/ens_svm.rda')

summary(ens_svm)
names(ens_svm)
names(ens_train_withtarget)
ens_validate_withtarget
ens_test_withtarget

ens_pred_SVM_train = predict(ens_svm,newdata=ens_train_withtarget, type="class")
ens_SVM_tr_table <- table(ens_pred_SVM_train, ens_train_withtarget$TARGET)
ens_accuracy_SVM_train <- sum(diag(ens_SVM_tr_table))/sum(ens_SVM_tr_table)*100
ens_accuracy_DT_train

ens_pred_SVM_validate = predict(ens_svm, newdata=ens_validate_withtarget, type="class")
ens_SVM_vatable <- table(ens_pred_SVM_validate, ens_validate_withtarget$TARGET)
accuracy_SVM_validate <- sum(diag(ens_SVM_vatable))/sum(ens_SVM_vatable)*100
accuracy_SVM_validate

ens_pred_SVM_test = predict(ens_svm, newdata=ens_test_withtarget, type="class")
ens_SVM_te_table <- table(ens_pred_SVM_test, HAR_test$Activity)
accuracy_SVM_test <- sum(diag(ens_SVM_te_table))/sum(ens_SVM_te_table)*100
accuracy_SVM_test
