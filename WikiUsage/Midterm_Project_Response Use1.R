################################################################################
## Midterm Project Part 2: Use logistic regression, LDA, and QAD to predict the 
## "Use1" behavior of Wikipedia based on the PCs (from part1) and the teachers'
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
wiki.Use1 <- wiki
varnames <- names(wiki[c(-1:-10)])

View(wiki.Use1)
NA.sum <- colSums(is.na(wiki.Use1))
NA.sum

## Using the df.na data frame as a placeholder to review the structure of the missing
## values
df.na <- subset(wiki.Use1, is.na(wiki.Use1$DOMAIN))
View(df.na)

for (v in varnames)
{
  wiki.Use1[, v] <- as.numeric(wiki.Use1[, v])
}

for (v in varnames)
{
  wiki.Use1[, v][is.na(wiki.Use1[, v])] <- median(wiki.Use1[,v], na.rm = TRUE)
}

wiki.Use1 <- wiki.Use1[c(-34, -35, -36, -37)] #data frame with only "Use1" as response

##Converting AGE and YEARSEXP from int to numeric type
wiki.Use1$AGE <- as.numeric(wiki.Use1$AGE)
wiki.Use1$YEARSEXP <- as.numeric(wiki.Use1$YEARSEXP)

## Deleting the two rows with NA for DOMAIN column since impossible to guess at the domain field
## Re-categorizing the DOMAIN levels into STEM and non-STEM 
wiki.Use1 <- wiki.Use1[c(-257, -295),]
wiki.Use1$FIELD <- ifelse(wiki.Use1$DOMAIN %in% c(2,3,4), "STEM", "nonSTEM")
wiki.Use1$FIELD <- as.factor(wiki.Use1$FIELD)


## Imputing missing values in YEARSEXP with mean of YEARSEXP of those rows where
## UOC_Position = c(2,3,6)
df1 <- subset(wiki.Use1, wiki.Use1$UOC_POSITION %in% c(2,3,6))
hist(df1$YEARSEXP, main="For UOC_Position = c(2,3,6)", ylab="Freq", xlab = paste("YEARSEXP" , sep=""))
wiki.Use1$YEARSEXP[is.na(wiki.Use1$YEARSEXP)] <- mean(df1$YEARSEXP, na.rm = TRUE)

## Checking proportion of registered wiki users and, imputing the four rows of
## missing values in USERWIKI to 0 (majority of the users are non-registered)
wikireg.vec <- wiki.Use1$USERWIKI
table(wikireg.vec)
wiki.Use1$USERWIKI[is.na(wiki.Use1$USERWIKI)] <- 0
wiki.Use1$WikiRegdUser <- as.factor(ifelse(wiki.Use1$USERWIKI == 0, "NO", "YES"))

##Remove the attribute column "OTHERSTATUS" - close to 60% of the rows are missing
## data, so using this column is impractical.
wiki.Use1 <- subset(wiki.Use1, select = -OTHERSTATUS)

## Creating alternate binary representation of the UOC_POSITION variable
df2 <- subset(wiki.Use1, !is.na(wiki.Use1$UOC_POSITION))
df2.phd <- subset(df2, df2$PhD == 1)
table(df2.phd$UOC_POSITION) # 103 tenure track, 243 non-tenure track
df2.NONphd <- subset(df2, df2$PhD == 0)
table(df2.NONphd$UOC_POSITION) # 18 tenure track, 434 non-tenure track
wiki.Use1["ACAD_TRACK"] <- NA
wiki.Use1$ACAD_TRACK[wiki.Use1$UOC_POSITION %in% c(1,2,3)] <- "TENURE"
wiki.Use1$ACAD_TRACK[wiki.Use1$UOC_POSITION %in% c(4,5,6)] <- "nonTENURE"
wiki.Use1$ACAD_TRACK[is.na(wiki.Use1$ACAD_TRACK) & wiki.Use1$PhD == 1] <- "TENURE"
wiki.Use1$ACAD_TRACK[is.na(wiki.Use1$ACAD_TRACK) & wiki.Use1$PhD == 0] <- "nonTENURE"
wiki.Use1 <- subset(wiki.Use1, select = -UOC_POSITION) #dropping UOC_Position variable
wiki.Use1$ACAD_TRACK <- as.factor(wiki.Use1$ACAD_TRACK)

##Remove the attribute column "OTHER_POSITION" - impossible to impute missing values (approx. 29% of the rows)
wiki.Use1 <- subset(wiki.Use1, select = -OTHER_POSITION)
##Remove the attribute column "DOMAIN" - alternate variable "FIELD" created
wiki.Use1 <- subset(wiki.Use1, select = -DOMAIN)

## Converting the remaining character variables into type factor
wiki.Use1$Gender <- as.factor(ifelse(wiki.Use1$GENDER == 0, "MALE", "FEMALE"))
wiki.Use1$Doctorate <- as.factor(ifelse(wiki.Use1$PhD == 0, "NO", "YES"))
wiki.Use1$University <- as.factor(ifelse(wiki.Use1$UNIVERSITY == 1, "UOC", "UPF"))

## Converting the Use1 ordinal variable into a binary categorical variable
wiki.Use1["TEACH_USE"] <- NA
wiki.Use1$TEACH_USE <- ifelse(wiki.Use1$Use1 %in% c(1,2), "NO", "YES")
wiki.Use1$TEACH_USE <- as.factor(wiki.Use1$TEACH_USE )

## Removing the redundant variables
wiki.Use1 <- subset(wiki.Use1, select = c(-GENDER, -PhD, -UNIVERSITY, -USERWIKI, -Use1))

table(wiki.Use1$TEACH_USE)
table(wiki.Use1$Gender)
table(wiki.Use1$Doctorate)
table(wiki.Use1$University)
table(wiki.Use1$WikiRegdUser)
table(wiki.Use1$FIELD)
table(wiki.Use1$ACAD_TRACK)


##Perform PCA to extract the principal components
wiki.Use1.pc <- subset(wiki.Use1, select = c(-AGE, -YEARSEXP, -FIELD, -WikiRegdUser, -ACAD_TRACK, -Gender,
                                             -Doctorate, -University, -TEACH_USE))
pca.out <- prcomp(wiki.Use1.pc, scale = TRUE)
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

for (i in 1:20)
{
  assign(paste("pc.comp",i,sep=""), 1*pc.comp[, i])
}

pc.data <- data.frame(pc.comp1, pc.comp2, pc.comp3, pc.comp4, pc.comp5, pc.comp6, pc.comp7,
                      pc.comp8, pc.comp9, pc.comp10, pc.comp11, pc.comp12, pc.comp13, pc.comp14,
                      pc.comp15, pc.comp16, pc.comp17, pc.comp18, pc.comp19, pc.comp20)

## Use cbind() to join the two data sets since rows are in the same order and 
## the number of records are the same.
wiki.Use1.logist <- cbind(wiki.Use1[c(-3:-40)], pc.data)

## Create stratified training and test data sets
set.seed(1)
train <- createDataPartition(paste(wiki.Use1.logist$Gender, wiki.Use1.logist$University,
                                   wiki.Use1.logist$WikiRegdUser, wiki.Use1.logist$FIELD, 
                                   wiki.Use1.logist$ACAD_TRACK, wiki.Use1.logist$TEACH_USE, 
                                   sep = ""), p = 0.5, list = FALSE)

train.data <- wiki.Use1.logist[train, ]
test.data <- wiki.Use1.logist[-train, ]
TEACH_USE.test <- wiki.Use1.logist$TEACH_USE[-train]


################################################################################
## Logistic Regression
################################################################################

model.logistic <- glm(TEACH_USE ~ ., data = wiki.Use1.logist, family = "binomial")
summary(model.logistic)
contrasts(wiki.Use1.logist$TEACH_USE)

## Logistic model regression with training data
logistic.fit <- glm(TEACH_USE ~ FIELD + pc.comp1 + pc.comp2 + pc.comp4 + pc.comp5 
                    + pc.comp7 + pc.comp10 + pc.comp12 + pc.comp13 + pc.comp17, 
                    data = train.data, family = "binomial")
summary(logistic.fit)

## Predict probability for "response" with test data
logistic.fit.probs <- predict(logistic.fit, test.data, type = "response")


## To predict "YES" for TEACH_USE by applying a 50% threshold to posterior probabilities
dim(test.data)
logistic.fit.pred <- rep("NO", 441)
logistic.fit.pred[logistic.fit.probs > 0.5] <- "YES"

## Produce the confusion matrix
table(logistic.fit.pred, TEACH_USE.test)

## Test error calculation: Fraction of responses for which wikipedia is used for teaching
mean(logistic.fit.pred == TEACH_USE.test)
mean(logistic.fit.pred != TEACH_USE.test)

## Create the ROC curve
library(pROC)
ROC <- roc(TEACH_USE.test, logistic.fit.probs)
plot(ROC, col = "blue", main = "ROC - Logistic Regression", cex.main =0.9,
     ylim = c(0, 1.02))

# Calculate the area under the curve (AUC)
auc(ROC)

################################################################################
## Linear Discriminant Analysis
################################################################################

lda.fit <- lda(TEACH_USE ~ FIELD + pc.comp1 + pc.comp2 + pc.comp4 + pc.comp5 + pc.comp7 
               + pc.comp10 + pc.comp12 + pc.comp13 + pc.comp17, data = train.data)
lda.fit 
plot(lda.fit, type = "both") 

## Predict "response" for the model based 
## on the linear discriminant function computations
lda.pred <- predict(lda.fit, test.data)
lda.class <-lda.pred$class
table(lda.class, TEACH_USE.test)
mean(lda.class == TEACH_USE.test)
mean(lda.class != TEACH_USE.test)

## Create the ROC curve
install.packages(("ROCR"))
library(ROCR)
par(mfrow=c(1,1))
prediction(lda.pred$posterior[, 2], TEACH_USE.test) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()
title(main = "ROC Curve - LDA")

##Compute AUC
prediction(lda.pred$posterior[, 2], TEACH_USE.test) %>%
  performance(measure = "auc") %>%
  .@y.values

################################################################################
## Quadratic Discriminant Analysis
################################################################################

qda.fit <- qda(TEACH_USE ~ FIELD + pc.comp1 + pc.comp2 + pc.comp4 
               + pc.comp5 + pc.comp7 + pc.comp10 + pc.comp12 + pc.comp13 + pc.comp17, 
               data = train.data)
qda.fit 

## Predict "response" for the model based on the quadratic discriminant functions
qda.pred <- predict(qda.fit, test.data)
qda.class <- qda.pred$class
table(qda.class, TEACH_USE.test)
mean(qda.class == TEACH_USE.test)
mean(qda.class != TEACH_USE.test)

## Create the ROC curve
install.packages(("ROCR"))
library(ROCR)
par(mfrow=c(1,1))
prediction(qda.pred$posterior[, 2], TEACH_USE.test) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()
title(main = "ROC Curve - QDA")

##Compute AUC
prediction(qda.pred$posterior[, 2], TEACH_USE.test) %>%
  performance(measure = "auc") %>%
  .@y.values
