#################################################################################
## Midterm Project Part1: Initial exploratory data analysis (EDA) of the data set 
#################################################################################
## Set significant digits upfront
options(digits=4)

## Load packages and libraries
install.packages("ggplot2")
install.packages("lattice")
install.packages("dplyr")
library(caret)
library(MASS)
library(dplyr)

## Read and view dataset
wiki = read.csv("wiki4HE.csv", header=T, sep=";", na.strings="?")
View(wiki)
str(wiki)
dim(wiki) #913 x 53
sum(is.na(wiki))

## Identifying missing data in each ordinal variable
NA.sum <- colSums(is.na(wiki))
NA.sum

## Selecting the survey ordinal variables (except for the Use variables) into a separate vector
vnames <- names(wiki[c(-1:-10, -33:-37)])

##Subset of original data set with only the survey original variables (except for Use variables) and missing values
## Setting the variables from integer to numeric type
wiki.copy <- wiki[c(-1:-10, -33:-37)]
for (v in vnames)
{
  wiki.copy[, v] <- as.numeric(wiki.copy[, v])
}
View(wiki.copy) 
sum(is.na(wiki.copy))

## Subset of original data set with ordinal variables (except for Use variables) 
## The missing values will be imputed to the median
wiki.medNA <- wiki[c(-1:-10, -33:-37)]
for (v in vnames)
{
  wiki.medNA[, v] <- as.numeric(wiki.medNA[, v]) # change from integer to numeric type
}

for (v in vnames)
{
  wiki.medNA[, v][is.na(wiki.medNA[, v])] <- median(wiki.medNA[,v], na.rm = TRUE)
}
View(wiki.medNA) 
sum(is.na(wiki.medNA))


## Plot univariate histograms
## Original data set
par(mfrow=c(3,3))
for (v in vnames) 
{
  hist(wiki[, v], main="", ylab="Freq", xlab = paste(v, sep=""))
}

par(mfrow=c(2,3))
hist(wiki$Use1, main="", ylab="Freq", xlab = paste("Use1", sep=""))
hist(wiki$Use2, main="", ylab="Freq", xlab = paste("Use2", sep=""))
hist(wiki$Use3, main="", ylab="Freq", xlab = paste("Use3", sep=""))
hist(wiki$Use4, main="", ylab="Freq", xlab = paste("Use4", sep=""))
hist(wiki$Use5, main="", ylab="Freq", xlab = paste("Use5", sep=""))

## New data set with missing values imputed to the median
par(mfrow=c(3,3))
for (v in vnames) 
{
  hist(wiki.medNA[, v], main="", ylab="Freq", xlab = paste(v, "_medNA" , sep=""))
}

## For plotting correlation matrix of the variables
install.packages("corrplot")
library(corrplot)

## Original data set
## The missing values are handled by case wise deletion by the cor() function
corr_matrix0 <- cor(wiki.copy, method = "spearman", use = "complete.obs")
par(mfrow=c(1,1))
corrplot(corr_matrix0, method = 'number', tl.cex=0.6, number.cex = 0.55, type = "lower",
         diag = FALSE,col = colorRampPalette(c("white", "deepskyblue", "blue4"))(100))
corrplot(corr_matrix0, method = 'color', tl.cex=0.6, type = "lower", diag = FALSE,
         title="Data set with original variables", mar=c(0,0,1,0), cex.main = 0.9)

#Data set in which missing values were imputed to the median 
corr_matrix1 <- cor(wiki.medNA, method = "spearman")
par(mfrow=c(1,1))
corrplot(corr_matrix1, method = 'number', tl.cex=0.6, number.cex = 0.55, type = "lower",
         diag = FALSE,col = colorRampPalette(c("white", "deepskyblue", "blue4"))(100))
corrplot(corr_matrix1, method = 'color', tl.cex=0.6, type = "lower", diag = FALSE,
         title="Data set with imputed variables", mar=c(0,0,1,0), cex.main = 0.9)


################################################################################
## Principal Component Analysis of the predictor variables - the ordinal survey 
## items (except for Use1 to Use 5)
################################################################################

## Perform principal component analysis of the ordinal 38 variables 
pr.out1 <- prcomp(wiki.medNA, scale = TRUE)
summary(pr.out1)

## Compute the proportion of variance for the original data set with missing values 
## imputed to median
pr.out1$rotation <- -pr.out1$rotation
pr.out1$x <- -pr.out1$x
pr.out1.var <- pr.out1$sdev^2
pve1 <- pr.out1.var/ sum(pr.out1.var)
par(mfrow=c(1,1))
## Plot proportion of variance explained versus components
plot(pve1, xlab = "Principal Components", ylab = "Proportion of Variance Explained", 
     main = "For data set with NA values imputed to median",type = "b", cex.main = 0.8)
## Calculate and plot the cumulative PVE
plot(cumsum(pve1), main="Cumulative PVE (NA values imputed to median)", xlab = "Principal Component", 
     ylab = "Cumulative Proportion of Variance Explained", ylim = c(0, 1), 
     type = "b", cex.axis = 1, cex.lab = 0.8, cex.main = 0.8, cex.sub = 0.6)
lines(cumsum(pve1))

## Install R-package for melt() function
install.packages("reshape")
library(reshape)

## Heatmap of survey items and first 10 PCs
melt_wiki.pca = melt(pr.out1$rotation[ ,c(1:10)])
colnames(melt_wiki.pca) = c("Survey_Items", "PC", "Value")
ggplot(data=melt_wiki.pca) + geom_tile(aes(x=PC, y=Survey_Items, fill=Value)) + 
  scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0) +
  labs(x="Principal component")


## Full spectrum of the survey item loadings in PC1, PC2 and PC3
melt_wiki.pca1 = melt(pr.out1$rotation[ ,c(1:3)])
colnames(melt_wiki.pca1) = c("Survey_Items", "PC", "Value")
ggplot(data=melt_wiki.pca1) + geom_col(aes(x=Survey_Items, y=Value)) + 
  facet_wrap(~PC, ncol=1) + theme(axis.text.x = element_text(angle = 90)) +
  labs(x="Survey_Items", y="Loading")


## Contribution of survey items on the first 10 principal components
pve1.mat = matrix(rep(pve1, each = 10), nrow=length(pve1))
survey_item.impact = apply(pr.out1$rotation[ ,c(1:10)]^2* pve1.mat, 1, sum)
melt_Survey = melt(survey_item.impact)
ggplot(data=melt_Survey) + 
  geom_col(aes(x=reorder(rownames(melt_Survey), -value), y=value)) + 
  theme(axis.text.x = element_text(angle = 90)) + 
  labs(x="Survey Items", y="Variable importance")



##################################################################################
## Exploratory of analysis of the response variables - Use1 to Use5
## and apply PCR to individual responses to determine optimal number of components
##################################################################################
library(pls)

wiki.medNA.pcr <- wiki[c(-1:-10)]
varnames <- names(wiki[c(-1:-10)])
for (var in varnames)
{
  wiki.medNA.pcr[, var] <- as.numeric(wiki.medNA.pcr[, var])
}

for (var in varnames)
{
  wiki.medNA.pcr[, var][is.na(wiki.medNA.pcr[, var])] <- median(wiki.medNA.pcr[,var], na.rm = TRUE)
}


#Plot univariate histograms
par(mfrow=c(3,3))
hist(wiki.medNA.pcr$Use1, main="", ylab="Freq", xlab = paste("Use1", sep=""))
hist(wiki.medNA.pcr$Use2, main="", ylab="Freq", xlab = paste("Use2", sep=""))
hist(wiki.medNA.pcr$Use3, main="", ylab="Freq", xlab = paste("Use3", sep=""))
hist(wiki.medNA.pcr$Use4, main="", ylab="Freq", xlab = paste("Use4", sep=""))
hist(wiki.medNA.pcr$Use5, main="", ylab="Freq", xlab = paste("Use5", sep=""))


corr_matrix2 <- cor(wiki.medNA.pcr[c(23:27)], method = "spearman")
par(mfrow=c(1,1))
corrplot(corr_matrix2, method = 'number', tl.cex=0.7, number.cex = 0.6,  
         title="Correlation of 'use' variables", mar=c(0,0,1,0), cex.main = 0.9)

wiki.medNA.pcr$recommendUse <- rowMeans(wiki.medNA.pcr[, c(25, 26)])
wiki.medNA.new <- wiki.medNA.pcr[c(-23:-26)]

## Create a training and test data sets for modeling "recommendUse"
set.seed(1)
wiki.medNA.new2 <- wiki.medNA.new[c(-23,-41)]
x2 <- model.matrix(recommendUse ~., wiki.medNA.new2)[, -1]#Creating a design X matrix, removing intercept
y2 <- wiki.medNA.new2$recommendUse
train2 <- sample(1:nrow(x2), 600)
test2 <- (-train2)
y.test2 <- y[test2]

## Perform PCR on the training data set and evaluate its test set performance
## for variable "recommendUse"
pcr.fit2 <- pcr(recommendUse ~., data = wiki.medNA.new2, subset = train2, scale = TRUE, 
                validation = "CV")
par(mfrow=c(1,1))
validationplot(pcr.fit2, val.type = "MSEP")
summary(pcr.fit2)
MSEP(pcr.fit2)
pcr.pred2 <- predict(pcr.fit2, x2[test2, ], ncomp = 16 )#Compute test MSE
mean((pcr.pred2 - y.test2)^2)

