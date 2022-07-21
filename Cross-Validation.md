### Intro

In this exercise, we will test the Cross Validation methodology to
evaluate the performance of 3 classification algorithms: **KNN**,
**Logistic Regression** and **Decision Tree**.

The Cross Validation methodology (**N-fold Cross Validation**)
partitions the date into **N** non-overlapping subsets to ensure that
every observation in the dataset is used on both training and test sets.

The dataset used contains information of patients who were diagnosed
with either a Malignant (M) or Benign (B) cancer.

#### 1. Importing libraries and dataset

``` r
library(ggplot2)
library(reshape2)
library(class)
library(gmodels)
library(caret)
library(party)

filePath <- "https://raw.githubusercontent.com/ggc0sta/tmu/main/data/prostate_cancer_dataset.csv"
pc_raw <- read.csv(filePath)
```

``` r
head(pc_raw)
```

    ##   id diagnosis_result radius texture perimeter area smoothness compactness
    ## 1  1                M     23      12       151  954      0.143       0.278
    ## 2  2                B      9      13       133 1326      0.143       0.079
    ## 3  3                M     21      27       130 1203      0.125       0.160
    ## 4  4                M     14      16        78  386      0.070       0.284
    ## 5  5                M      9      19       135 1297      0.141       0.133
    ## 6  6                B     25      25        83  477      0.128       0.170
    ##   symmetry fractal_dimension
    ## 1    0.242             0.079
    ## 2    0.181             0.057
    ## 3    0.207             0.060
    ## 4    0.260             0.097
    ## 5    0.181             0.059
    ## 6    0.209             0.076

``` r
str(pc_raw)
```

    ## 'data.frame':    100 obs. of  10 variables:
    ##  $ id               : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ diagnosis_result : chr  "M" "B" "M" "M" ...
    ##  $ radius           : int  23 9 21 14 9 25 16 15 19 25 ...
    ##  $ texture          : int  12 13 27 16 19 25 26 18 24 11 ...
    ##  $ perimeter        : int  151 133 130 78 135 83 120 90 88 84 ...
    ##  $ area             : int  954 1326 1203 386 1297 477 1040 578 520 476 ...
    ##  $ smoothness       : num  0.143 0.143 0.125 0.07 0.141 0.128 0.095 0.119 0.127 0.119 ...
    ##  $ compactness      : num  0.278 0.079 0.16 0.284 0.133 0.17 0.109 0.165 0.193 0.24 ...
    ##  $ symmetry         : num  0.242 0.181 0.207 0.26 0.181 0.209 0.179 0.22 0.235 0.203 ...
    ##  $ fractal_dimension: num  0.079 0.057 0.06 0.097 0.059 0.076 0.057 0.075 0.074 0.082 ...

#### 2. Data transformation

-   Remove the **id** variable.
-   Normalize the dataset to transfer the variable values to a common
    scale.
-   Convert the **diagnosis_result** variable to factor.

``` r
pc <- pc_raw[,-1]
pcNorm <- pc


for(i in colnames(pcNorm)){
  if((class(pcNorm[,i]) == "numeric") | (class(pcNorm[,i]) == "integer")){
      x <- pcNorm[,i]
      pcNorm[,i] <- (x - min(x)) / (max(x) - min(x))    
  }

}

pcNorm$diagnosis_result <- factor(pcNorm$diagnosis_result)
```

#### 3. Cross Validation

``` r
set.seed(1)

folds <- createFolds(pcNorm$diagnosis_result, k = 10)

fAccuracy <- data.frame(knn = numeric(10), log = numeric(10), dt = numeric(10))

for(i in 1:10){
  
  f <- folds[[i]]
  
  # 1. Dividing training & test sets
  train <- pcNorm[-f,]   
  test <- pcNorm[f,]
  
  ## i. storing the labels
  lblTrain <- train$diagnosis_result
  lblTest <- test$diagnosis_result
  
  ## ii. removing labels from train & test sets
  train_noLbl <- train[-1]
  test_noLbl <- test[-1]
  
  # 2. Modeling
  knnPredictions <- knn(train = train_noLbl, test = test_noLbl, cl = lblTrain, k = 3)
  glm1 <- glm(diagnosis_result ~., data = train, family = "binomial")
  dt <- ctree(diagnosis_result ~., data = train)
  
  # 3. Storing accuracy results
  ## i. knn
  cm <- table(lblTest, knnPredictions)
  acc <- confusionMatrix(cm, positive = "M")$overall["Accuracy"]
  fAccuracy$knn[i] <- acc 
  
  ## ii. logistic reg
  pred <- predict(glm1, newdata = test)
  pred <- ifelse(pred >= 0.5, 1, 0)
  pred <- factor(pred, labels = c("B", "M"))
  
  cm <- table(pred, test$diagnosis_result)
  acc <- confusionMatrix(cm)$overall["Accuracy"]
  fAccuracy$log[i] <- acc 
  
  ## iii. decision tree
  dtPredictions <- predict(dt, test)
  cm <- table(lblTest, dtPredictions)
  acc <- confusionMatrix(cm, positive = "M")$overall["Accuracy"]
  fAccuracy$dt[i] <- acc
}
```

#### 4. Model Evaluation

``` r
A <- melt(fAccuracy)
ggplot(A, aes(y=value, x=variable, color=variable)) +
  geom_boxplot(alpha = 0.5)
```

![](Cross-Validation_files/figure-markdown_github/unnamed-chunk-6-1.png)

To check if there is a difference in model performance, we perform the
Non-parametric Kruskal-Wallis test. Given the p-value of the test, we do
not reject the null hypothesis that there is a difference in model
performance.

``` r
# Kruskal-Wallis test
# Hypothesis testing for more than two non-parametric variables

# H0: k populations are identical
# H1: at least 2 of the k populations differ

kruskal.test(formula = value ~ variable, data = A) 
```

    ## 
    ##  Kruskal-Wallis rank sum test
    ## 
    ## data:  value by variable
    ## Kruskal-Wallis chi-squared = 1.2961, df = 2, p-value = 0.5231
