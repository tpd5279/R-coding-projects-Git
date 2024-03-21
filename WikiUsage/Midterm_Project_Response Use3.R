################################################################################
## Midterm Project Part 2: Use logistic regression, LDA, and QAD to predict the 
## "Use3" behavior of Wikipedia based on the PCs (from part1) and the teachers'
## attributes
################################################################################
## Set significant digits upfront
options(digits=4)

## Load packages and libraries
install.packages("ggplot2")
install.packages("lattice")
install.packages("dplyr")
library(caret)
library(MASS)
library(dplyr)

## Read the dataset
wiki = read.csv("wiki4HE.csv", header=T, sep=";", na.strings="?")

##Pre-processing the data set 
wiki.Use3 <- wiki
varnames <- names(wiki[c(-1:-10)])

View(wiki.Use3)
NA.sum <- colSums(is.na(wiki.Use3))
NA.sum

## Using the df.na data frame as a placeholder to review the structure of the missing
## values
df.na <- subset(wiki.Use3, is.na(wiki.Use3$UOC_POSITION))
View(df.na)

for (v in varnames)
{
  wiki.Use3[, v] <- as.numeric(wiki.Use3[, v])
}

for (v in varnames)
{
  wiki.Use3[, v][is.na(wiki.Use3[, v])] <- median(wiki.Use3[,v], na.rm = TRUE)
}

wiki.Use3 <- wiki.Use3[c(-33, -34, -36, -37)] #data frame with only "Use3" as response

##Converting AGE and YEARSEXP from int to numeric type
wiki.Use3$AGE <- as.numeric(wiki.Use3$AGE)
wiki.Use3$YEARSEXP <- as.numeric(wiki.Use3$YEARSEXP)

## Deleting the two rows with NA for DOMAIN column and re-categorizing
## the DOMAIN levels into STEM and non-STEM 
wiki.Use3 <- wiki.Use3[c(-257, -295),]
wiki.Use3$FIELD <- ifelse(wiki.Use3$DOMAIN %in% c(2,3,4), "STEM", "nonSTEM")
wiki.Use3$FIELD <- as.factor(wiki.Use3$FIELD)

## Imputing missing values in YEARSEXP with mean of YEARSEXP of those rows where
## UOC_Position = c(2,3,6)
df1 <- subset(wiki.Use3, wiki.Use3$UOC_POSITION %in% c(2,3,6))
wiki.Use3$YEARSEXP[is.na(wiki.Use3$YEARSEXP)] <- mean(df1$YEARSEXP, na.rm = TRUE)

## Missing values in USERWIKI to 0 (majority of the users are non-registered)
wiki.Use3$USERWIKI[is.na(wiki.Use3$USERWIKI)] <- 0
wiki.Use3$WikiRegdUser <- as.factor(ifelse(wiki.Use3$USERWIKI == 0, "NO", "YES"))

## Creating alternate binary representation of the UOC_POSITION variable
wiki.Use3["ACAD_TRACK"] <- NA
wiki.Use3$ACAD_TRACK[wiki.Use3$UOC_POSITION %in% c(1,2,3)] <- "TENURE"
wiki.Use3$ACAD_TRACK[wiki.Use3$UOC_POSITION %in% c(4,5,6)] <- "nonTENURE"
wiki.Use3$ACAD_TRACK[is.na(wiki.Use3$ACAD_TRACK) & wiki.Use3$PhD == 1] <- "TENURE"
wiki.Use3$ACAD_TRACK[is.na(wiki.Use3$ACAD_TRACK) & wiki.Use3$PhD == 0] <- "nonTENURE"
wiki.Use3$ACAD_TRACK <- as.factor(wiki.Use3$ACAD_TRACK)

## Converting the remaining character variables into type factor
wiki.Use3$Gender <- as.factor(ifelse(wiki.Use3$GENDER == 0, "MALE", "FEMALE"))
wiki.Use3$Doctorate <- as.factor(ifelse(wiki.Use3$PhD == 0, "NO", "YES"))
wiki.Use3$University <- as.factor(ifelse(wiki.Use3$UNIVERSITY == 1, "UOC", "UPF"))

## Converting the Use3 ordinal variable into a binary categorical variable
wiki.Use3["RECOMMEND_USE"] <- NA
wiki.Use3$RECOMMEND_USE <- ifelse(wiki.Use3$Use3 %in% c(1,2), "NO", "YES")
wiki.Use3$RECOMMEND_USE <- as.factor(wiki.Use3$RECOMMEND_USE)

## Removing the redundant variables
wiki.Use3 <- subset(wiki.Use3, select = c(-GENDER, -PhD, -UNIVERSITY, -USERWIKI, -Use3,
                                          -OTHER_POSITION, -DOMAIN, -OTHERSTATUS, -UOC_POSITION))

table(wiki.Use3$RECOMMEND_USE)
table(wiki.Use3$Gender)
table(wiki.Use3$Doctorate)
table(wiki.Use3$University)
table(wiki.Use3$WikiRegdUser)
table(wiki.Use3$FIELD)
table(wiki.Use3$ACAD_TRACK)

##Perform PCA to extract the principal components
wiki.Use3.pc <- subset(wiki.Use3, select = c(-AGE, -YEARSEXP, -FIELD, -WikiRegdUser, 
                                             -ACAD_TRACK, -Gender, -Doctorate, -University,
                                             -RECOMMEND_USE))
pca.out <- prcomp(wiki.Use3.pc, scale = TRUE)
summary(pca.out)
pca.out$rotation <- -pca.out$rotation
pca.out$x <- -pca.out$x
pca.out.var <- pca.out$sdev^2
pve <- pca.out.var/ sum(pca.out.var)
par(mfrow=c(1,1))
## Plot proportion of variance explained versus components
plot(pve, xlab = "Principal Components", ylab = "Proportion of Variance Explained", 
     main = "For data set with NA values imputed by median",type = "b", cex.main = 0.8)
pc.comp <- pca.out$x

for (i in 1:16)
{
  assign(paste("pc.comp",i,sep=""), 1*pc.comp[, i])
}

pc.data <- data.frame(pc.comp1, pc.comp2, pc.comp3, pc.comp4, pc.comp5, pc.comp6, pc.comp7,
                      pc.comp8, pc.comp9, pc.comp10, pc.comp11, pc.comp12, pc.comp13, pc.comp14,
                      pc.comp15, pc.comp16)

## Use cbind() to join the two data sets since rows are in the same order and 
## the number of records are the same.
wiki.Use3.logist <- cbind(wiki.Use3[c(-3:-40)], pc.data)

## Create stratified training and test data sets
set.seed(1)
train <- createDataPartition(paste(wiki.Use3.logist$Gender, wiki.Use3.logist$University,
                                   wiki.Use3.logist$WikiRegdUser, wiki.Use3.logist$FIELD, 
                                   wiki.Use3.logist$ACAD_TRACK, sep = ""), 
                             p = 0.5, list = FALSE)

train.data <- wiki.Use3.logist[train, ]
test.data <- wiki.Use3.logist[-train, ]
RECOMMEND_USE.test <- wiki.Use3.logist$RECOMMEND_USE[-train]


################################################################################
## Logistic Regression
################################################################################

model.logistic <- glm(RECOMMEND_USE ~ ., data = wiki.Use3.logist, family = "binomial")
summary(model.logistic)
contrasts(wiki.Use3.logist$RECOMMEND_USE)

## Logistic model regression with training data
logistic.fit <- glm(RECOMMEND_USE ~ pc.comp1 + pc.comp2 + pc.comp4 + pc.comp6 + pc.comp7
                    + pc.comp8 + pc.comp9 + pc.comp11 + pc.comp12 + pc.comp13 + pc.comp14, 
                    data = train.data, family = "binomial")
summary(logistic.fit)

## Predict probability for "response" with test data
logistic.fit.probs <- predict(logistic.fit, test.data, type = "response")


## To predict "YES" for RECOMMEND_USE by applying a 50% threshold to posterior probabilities
dim(test.data)
logistic.fit.pred <- rep("NO", 444)
logistic.fit.pred[logistic.fit.probs > 0.5] <- "YES"

## Produce the confusion matrix
table(logistic.fit.pred, RECOMMEND_USE.test)

## Test error calculation: Fraction of responses for which wikipedia is recommended
mean(logistic.fit.pred == RECOMMEND_USE.test)
mean(logistic.fit.pred != RECOMMEND_USE.test)

## Create the ROC curve
library(pROC)
ROC <- roc(RECOMMEND_USE.test,logistic.fit.probs)
plot(ROC, col = "blue", main = "ROC - Logistic Regression", cex.main =0.9,
     ylim = c(0, 1.02))

# Calculate the area under the curve (AUC)
auc(ROC)


################################################################################
## Linear Discriminant Analysis
################################################################################

lda.fit <- lda(RECOMMEND_USE ~ pc.comp1 + pc.comp2 + pc.comp4 + pc.comp6 + pc.comp7
               + pc.comp8 + pc.comp9 + pc.comp11 + pc.comp12 + pc.comp13 + pc.comp14, 
               data = train.data)
lda.fit 
plot(lda.fit, type = "both") 

## Predict "response" for the model based 
## on the linear discriminant function computations
lda.pred <- predict(lda.fit, test.data)
lda.class <-lda.pred$class
table(lda.class, RECOMMEND_USE.test)
mean(lda.class == RECOMMEND_USE.test)
mean(lda.class != RECOMMEND_USE.test)
sum(lda.pred$posterior[, 2] > 0.5)

## Create the ROC curve
install.packages(("ROCR"))
library(ROCR)
par(mfrow=c(1,1))
prediction(lda.pred$posterior[, 2], RECOMMEND_USE.test) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()
title(main = "ROC Curve - LDA")

##Compute AUC
prediction(lda.pred$posterior[, 2], RECOMMEND_USE.test) %>%
  performance(measure = "auc") %>%
  .@y.values


################################################################################
## Quadratic Discriminant Analysis
################################################################################

qda.fit <- qda(RECOMMEND_USE ~ pc.comp1 + pc.comp2 + pc.comp4 + pc.comp6 + pc.comp7
               + pc.comp8 + pc.comp9 + pc.comp11 + pc.comp12 + pc.comp13 + pc.comp14, 
               data = train.data)
qda.fit 

## Predict "response" for the model based on the quadratic discriminant functions
qda.pred <- predict(qda.fit, test.data)
qda.class <- qda.pred$class
table(qda.class, RECOMMEND_USE.test)
mean(qda.class == RECOMMEND_USE.test)
mean(qda.class != RECOMMEND_USE.test)
sum(qda.pred$posterior[, 2] > 0.5)

## Create the ROC curve
install.packages(("ROCR"))
library(ROCR)
par(mfrow=c(1,1))
prediction(qda.pred$posterior[, 2], RECOMMEND_USE.test) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()
title(main = "ROC Curve - QDA")

##Compute AUC
prediction(qda.pred$posterior[, 2], RECOMMEND_USE.test) %>%
  performance(measure = "auc") %>%
  .@y.values

