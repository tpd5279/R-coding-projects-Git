################################################################################
## Midterm Project Part 2: Use logistic regression, LDA, and QAD to predict the 
## "Use5" behavior of Wikipedia based on the PCs (from part1) and the teachers'
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
wiki.Use5 <- wiki
varnames <- names(wiki[c(-1:-10)])

View(wiki.Use5)
NA.sum <- colSums(is.na(wiki.Use5))
NA.sum

## Using the df.na data frame as a placeholder to review the structure of the missing
## values
df.na <- subset(wiki.Use5, is.na(wiki.Use5$DOMAIN))
View(df.na)

for (v in varnames)
{
  wiki.Use5[, v] <- as.numeric(wiki.Use5[, v])
}

for (v in varnames)
{
  wiki.Use5[, v][is.na(wiki.Use5[, v])] <- median(wiki.Use5[,v], na.rm = TRUE)
}

wiki.Use5 <- wiki.Use5[c(-33, -34, -35, -36)] #data frame with only "Use5" as response

##Converting AGE and YEARSEXP from int to numeric type
wiki.Use5$AGE <- as.numeric(wiki.Use5$AGE)
wiki.Use5$YEARSEXP <- as.numeric(wiki.Use5$YEARSEXP)

## Deleting the two rows with NA for DOMAIN column and re-categorizing
## the DOMAIN levels into STEM and non-STEM 
wiki.Use5 <- wiki.Use5[c(-257, -295),]
wiki.Use5$FIELD <- ifelse(wiki.Use5$DOMAIN %in% c(2,3,4), "STEM", "nonSTEM")
wiki.Use5$FIELD <- as.factor(wiki.Use5$FIELD)

## Imputing missing values in YEARSEXP with mean of YEARSEXP of those rows where
## UOC_Position = c(2,3,6)
df1 <- subset(wiki.Use5, wiki.Use5$UOC_POSITION %in% c(2,3,6))
wiki.Use5$YEARSEXP[is.na(wiki.Use5$YEARSEXP)] <- mean(df1$YEARSEXP, na.rm = TRUE)

## Missing values in USERWIKI to 0 (majority of the users are non-registered)
wiki.Use5$USERWIKI[is.na(wiki.Use5$USERWIKI)] <- 0
wiki.Use5$WikiRegdUser <- as.factor(ifelse(wiki.Use5$USERWIKI == 0, "NO", "YES"))

##Remove the attribute column "OTHERSTATUS" 
wiki.Use5 <- subset(wiki.Use5, select = -OTHERSTATUS)

## Creating alternate binary representation of the UOC_POSITION variable
wiki.Use5["ACAD_TRACK"] <- NA
wiki.Use5$ACAD_TRACK[wiki.Use5$UOC_POSITION %in% c(1,2,3)] <- "TENURE"
wiki.Use5$ACAD_TRACK[wiki.Use5$UOC_POSITION %in% c(4,5,6)] <- "nonTENURE"
wiki.Use5$ACAD_TRACK[is.na(wiki.Use5$ACAD_TRACK) & wiki.Use5$PhD == 1] <- "TENURE"
wiki.Use5$ACAD_TRACK[is.na(wiki.Use5$ACAD_TRACK) & wiki.Use5$PhD == 0] <- "nonTENURE"
wiki.Use5 <- subset(wiki.Use5, select = -UOC_POSITION) #dropping UOC_Position variable
wiki.Use5$ACAD_TRACK <- as.factor(wiki.Use5$ACAD_TRACK)

##Remove the attribute column "OTHER_POSITION" 
wiki.Use5 <- subset(wiki.Use5, select = -OTHER_POSITION)
##Remove the attribute column "DOMAIN" - alternate variable "FIELD" created
wiki.Use5 <- subset(wiki.Use5, select = -DOMAIN)

## Converting the remaining character variables into type factor
wiki.Use5$Gender <- as.factor(ifelse(wiki.Use5$GENDER == 0, "MALE", "FEMALE"))
wiki.Use5$Doctorate <- as.factor(ifelse(wiki.Use5$PhD == 0, "NO", "YES"))
wiki.Use5$University <- as.factor(ifelse(wiki.Use5$UNIVERSITY == 1, "UOC", "UPF"))

## Converting the Use5 ordinal variable into a binary categorical variable
wiki.Use5["PERCEPTION_USE"] <- NA
wiki.Use5$PERCEPTION_USE <- ifelse(wiki.Use5$Use5 %in% c(1,2), "NO", "YES")
wiki.Use5$PERCEPTION_USE <- as.factor(wiki.Use5$PERCEPTION_USE)

## Removing the redundant variables
wiki.Use5 <- subset(wiki.Use5, select = c(-GENDER, -PhD, -UNIVERSITY, -USERWIKI, -Use5))

table(wiki.Use5$PERCEPTION_USE)
table(wiki.Use5$Gender)
table(wiki.Use5$Doctorate)
table(wiki.Use5$University)
table(wiki.Use5$WikiRegdUser)
table(wiki.Use5$FIELD)
table(wiki.Use5$ACAD_TRACK)

##Perform PCA to extract the principal components
wiki.Use5.pc <- subset(wiki.Use5, select = c(-AGE, -YEARSEXP, -FIELD, -WikiRegdUser, 
                                             -ACAD_TRACK, -Gender, -Doctorate, -University,
                                             -PERCEPTION_USE))
pca.out <- prcomp(wiki.Use5.pc, scale = TRUE)
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

for (i in 1:19)
{
  assign(paste("pc.comp",i,sep=""), 1*pc.comp[, i])
}

pc.data <- data.frame(pc.comp1, pc.comp2, pc.comp3, pc.comp4, pc.comp5, pc.comp6, pc.comp7,
                      pc.comp8, pc.comp9, pc.comp10, pc.comp11, pc.comp12, pc.comp13, 
                      pc.comp14,pc.comp15, pc.comp16, pc.comp17, pc.comp18, pc.comp19)

## Use cbind() to join the two data sets since rows are in the same order and 
## the number of records are the same.
wiki.Use5.logist <- cbind(wiki.Use5[c(-3:-40)], pc.data)

## Create stratified training and test data sets
set.seed(1)
train <- createDataPartition(paste(wiki.Use5.logist$Gender, wiki.Use5.logist$University,
                                   wiki.Use5.logist$WikiRegdUser, wiki.Use5.logist$FIELD, 
                                   wiki.Use5.logist$ACAD_TRACK, wiki.Use5.logist$PERCEPTION_USE, 
                                   sep = ""), p = 0.5, list = FALSE)

train.data <- wiki.Use5.logist[train, ]
test.data <- wiki.Use5.logist[-train, ]
PERCEPTION_USE.test <- wiki.Use5.logist$PERCEPTION_USE[-train]

################################################################################
## Logistic Regression
################################################################################

model.logistic <- glm(PERCEPTION_USE ~ ., data = wiki.Use5.logist, family = "binomial")
summary(model.logistic)
contrasts(wiki.Use5.logist$PERCEPTION_USE)

## Logistic model regression with training data
logistic.fit <- glm(PERCEPTION_USE ~ pc.comp1 + pc.comp2 + pc.comp3 + pc.comp5 + pc.comp6
                    + pc.comp7 + pc.comp8 + pc.comp9 + pc.comp13 + pc.comp16 + pc.comp18, 
                    data = train.data, family = "binomial")
summary(logistic.fit)

## Predict probability for "response" with test data
logistic.fit.probs <- predict(logistic.fit, test.data, type = "response")

## To predict "YES" for PERCEPTION_USE by applying a 50% threshold to posterior probabilities
dim(test.data)
logistic.fit.pred <- rep("NO", 443)
logistic.fit.pred[logistic.fit.probs > 0.5] <- "YES"

## Produce the confusion matrix
table(logistic.fit.pred, PERCEPTION_USE.test)

## Test error calculation: Fraction of responses for which wikipedia is perceived 
## to be used by students
mean(logistic.fit.pred == PERCEPTION_USE.test)
mean(logistic.fit.pred != PERCEPTION_USE.test)

## Create the ROC curve
library(pROC)
ROC <- roc(PERCEPTION_USE.test, logistic.fit.probs)
plot(ROC, col = "blue", main = "ROC - Logistic Regression", cex.main =0.9,
     ylim = c(0, 1.02))

# Calculate the area under the curve (AUC)
auc(ROC)


################################################################################
## Linear Discriminant Analysis
################################################################################

lda.fit <- lda(PERCEPTION_USE ~ pc.comp1 + pc.comp2 + pc.comp3 + pc.comp5 + pc.comp6 
               + pc.comp7 + pc.comp8 + pc.comp9 + pc.comp13 + pc.comp16 + pc.comp18, 
               data = train.data)
lda.fit 
plot(lda.fit, type = "both") 

## Predict "response" for the model based 
## on the linear discriminant function computations
lda.pred <- predict(lda.fit, test.data)
lda.class <-lda.pred$class
table(lda.class, PERCEPTION_USE.test)
mean(lda.class == PERCEPTION_USE.test)
mean(lda.class != PERCEPTION_USE.test)

## Create the ROC curve
install.packages(("ROCR"))
library(ROCR)
par(mfrow=c(1,1))
prediction(lda.pred$posterior[, 2], PERCEPTION_USE.test) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()
title(main = "ROC Curve - LDA")

##Compute AUC
prediction(lda.pred$posterior[, 2], PERCEPTION_USE.test) %>%
  performance(measure = "auc") %>%
  .@y.values


################################################################################
## Quadratic Discriminant Analysis
################################################################################

qda.fit <- qda(PERCEPTION_USE ~ pc.comp1 + pc.comp2 + pc.comp3 + pc.comp5 + pc.comp6 
               + pc.comp7 + pc.comp8 + pc.comp9 + pc.comp13 + pc.comp16 + pc.comp18, 
               data = train.data)
qda.fit 

## Predict "response" for the model based on the quadratic discriminant functions
qda.pred <- predict(qda.fit, test.data)
qda.class <- qda.pred$class
table(qda.class, PERCEPTION_USE.test)
mean(qda.class == PERCEPTION_USE.test)
mean(qda.class != PERCEPTION_USE.test)

## Create the ROC curve
install.packages(("ROCR"))
library(ROCR)
par(mfrow=c(1,1))
prediction(qda.pred$posterior[, 2], PERCEPTION_USE.test) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()
title(main = "ROC Curve - QDA")

##Compute AUC
prediction(qda.pred$posterior[, 2], PERCEPTION_USE.test) %>%
  performance(measure = "auc") %>%
  .@y.values












