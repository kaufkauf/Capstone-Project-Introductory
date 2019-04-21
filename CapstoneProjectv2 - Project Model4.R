library(data.table)
library(dplyr)
library(caret)
library(ROCR)

#read in data sets
setwd('C:/Users/kkaufman/Documents/Data Sci/Capstone/UCI/UCI bank files/bank-additional')
bankfull <- fread('bank-additional-full.csv')

#take a look at the data and summary statistics
glimpse(bankfull)
summary(bankfull)

#transform to data frame
bankfull <- as.data.frame(bankfull)

#functions for outlier imputation
lowerbound <- function(x){
  y1 <- quantile(x, .05, names = FALSE)
  return(y1)
}

upperbound <- function(x){
  y2 <- quantile(x, .95, names = FALSE)
  return(y2)
}

#manual outlier imputation
bankfull <- bankfull %>%
  mutate(age = ifelse(age < lowerbound(age), lowerbound(age), ifelse(age > upperbound(age), upperbound(age), age))) %>%
  mutate(duration = ifelse(duration < lowerbound(duration), lowerbound(duration), ifelse(duration > upperbound(duration), upperbound(duration), duration))) %>%
  mutate(campaign = ifelse(campaign < lowerbound(campaign), lowerbound(campaign), ifelse(campaign > upperbound(campaign), upperbound(campaign), campaign))) %>%
  mutate(pdays = ifelse(pdays < lowerbound(pdays), lowerbound(pdays), ifelse(pdays > upperbound(pdays), upperbound(pdays), pdays))) %>%
  mutate(previous = ifelse(previous < lowerbound(previous), lowerbound(previous), ifelse(previous > upperbound(previous), upperbound(previous), previous))) %>%
  mutate(emp.var.rate = ifelse(emp.var.rate < lowerbound(emp.var.rate), lowerbound(emp.var.rate), ifelse(emp.var.rate > upperbound(emp.var.rate), upperbound(emp.var.rate), emp.var.rate))) %>%
  mutate(cons.price.idx = ifelse(cons.price.idx < lowerbound(cons.price.idx), lowerbound(cons.price.idx), ifelse(cons.price.idx > upperbound(cons.price.idx), upperbound(cons.price.idx), cons.price.idx))) %>%
  mutate(cons.conf.idx = ifelse(cons.conf.idx < lowerbound(cons.conf.idx), lowerbound(cons.conf.idx), ifelse(cons.conf.idx > upperbound(cons.conf.idx), upperbound(cons.conf.idx), cons.conf.idx))) %>%
  mutate(euribor3m = ifelse(euribor3m < lowerbound(euribor3m), lowerbound(euribor3m), ifelse(euribor3m > upperbound(euribor3m), upperbound(euribor3m), euribor3m))) %>%
  mutate(nr.employed = ifelse(nr.employed < lowerbound(nr.employed), lowerbound(nr.employed), ifelse(nr.employed > upperbound(nr.employed), upperbound(nr.employed), nr.employed)))

#group categoricals & convert to factors
bankfull <- bankfull %>%
  mutate(job = as.factor(ifelse(job %in% c("housemaid","management","entrepreneur","retired","self-employed","student","unemployed","unknown"),"other",job))) %>%
  mutate(marital = as.factor(marital)) %>%
  mutate(education = as.factor(ifelse(education %in% c("illiterate","unknown","basic.6y","basic.4y"),"other",education))) %>%
  mutate(default = as.factor(default)) %>%
  mutate(housing = as.factor(housing)) %>%
  mutate(loan = as.factor(loan)) %>%
  mutate(contact = as.factor(contact)) %>%
  mutate(month = as.factor(ifelse(month %in% c("dec","sep","mar","oct","apr","nov"),"other",month))) %>%
  mutate(day_of_week = as.factor(day_of_week)) %>%
  mutate(poutcome = as.factor(poutcome)) %>%
  mutate(y = as.factor(y))
  
#look for zero variance/near zero variance & save results
nzv <- nearZeroVar(bankfull, saveMetrics= TRUE)  #pdays has an NZV with a freqRatio of 90.37 & a percentUnique of 0.065

#remove nzv
bankfull <- filter(bankfull[, which(!colnames(bankfull) %in% c("pdays"))])

#find highly correlated variables and remove
corr.x <- findCorrelation(cor(bankfull[, which(!colnames(bankfull) %in% c("job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome","y"))]), cutoff = 0.5, names = TRUE)
df1 <- bankfull[, which(!colnames(bankfull) %in% c(corr.x))]

#data partitioning at 10% test / 90% train
set.seed(88)
trainIndex <- createDataPartition(df1$y, p = .9, 
                                  list = FALSE, 
                                  times = 1)
bankTrain <- df1[ trainIndex,]
bankTest <- df1[-trainIndex,]

# set control for sample model
myControl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  verboseIter = TRUE
)

#build model with control
mod1 <- train(factor(y) ~ ., data = bankTrain, method = "glm", trControl = myControl, metric="ROC")

#make predictions on test data using mod1 & look at structure of results
p <- predict(mod1, bankTest, type = "prob")
hist(p$yes)
table(bankTest$y)

#plot ROC curve
predobject <- ROCR::prediction(p$yes, bankTest$y)
rocobject <- ROCR::performance(predobject,  "tpr", "fpr")
plot(rocobject, colorize=TRUE)
#recommended threshold value at 0.2
auc.value <- ROCR::performance(predobject, "auc")@y.values[[1]]
#save area under the curve - .9114

#plot precision/recall curve
probject<- ROCR::performance(predobject, "prec", "rec")
plot(probject, colorize=TRUE)
abline(h = 0.6)
#recommended threshold value that gives a precision of 0.6 and a recall of 0.4 is 0.5

#implement 0.2 threshold & save confusion matrix
confmatrix <- table(bankTest$y, p$yes < 0.2)
