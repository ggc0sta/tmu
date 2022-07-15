------------------------------------------------------------------------

Using data from Yelp to predict the probability of a restaurant having
an above average rating.

------------------------------------------------------------------------

##### 1. Importing required libraries and data set.

Reference to the original data: <https://www.yelp.com/dataset>

``` r
library(ggplot2)
library(caret)
```

``` r
yelp <- read.csv("https://raw.githubusercontent.com/ggc0sta/tmu/main/data/yelp_dataset.csv",
                 stringsAsFactors = TRUE)
yelp <- yelp[,!(names(yelp)  %in% c("id"))] #removing id column
```

------------------------------------------------------------------------

##### 2. Describing data

-   **city** - Location of each restaurant. Toronto, Markham or
    Mississauga
-   **class** - Above average rating (0 = no, 1 = yes)
-   **review_count** - number of reviews in yelp
-   **categories** - type of restaurant

``` r
summary(yelp) #no missing values
```

    ##          city          class         review_count    
    ##  Markham   : 760   Min.   :0.0000   Min.   :   3.00  
    ##  Missisauga:1309   1st Qu.:0.0000   1st Qu.:   7.00  
    ##  Toronto   :7150   Median :0.0000   Median :  17.00  
    ##                    Mean   :0.3405   Mean   :  39.71  
    ##                    3rd Qu.:1.0000   3rd Qu.:  45.00  
    ##                    Max.   :1.0000   Max.   :1494.00  
    ##                                                      
    ##                 categories  
    ##  Asian               :2276  
    ##  Fast Food           :1528  
    ##  Middle Eastern      :1187  
    ##  Coffee or Sandwiches:1147  
    ##  North American      : 936  
    ##  Other               : 802  
    ##  (Other)             :1343

``` r
head(yelp)
```

    ##      city class review_count           categories
    ## 1 Toronto     0           12              Italian
    ## 2 Toronto     0           39                  Pub
    ## 3 Toronto     1            3 Coffee or Sandwiches
    ## 4 Toronto     1           55       Middle Eastern
    ## 5 Markham     0           80                Asian
    ## 6 Toronto     0            5                Asian

------------------------------------------------------------------------

##### 3. Preparing the data for modeling.

``` r
set.seed(1) #for reproducibility
sampleSize <- floor(0.8*nrow(yelp))
trainIndex <- sample(1:nrow(yelp), size = sampleSize)

train <- yelp[trainIndex,]
test <- yelp[-trainIndex,]

# set sizes
dim(train);dim(test)
```

    ## [1] 7375    4

    ## [1] 1844    4

##### 4. Creating a Logistic Regression model.

The **glm** function converts the categorical variables **city** and
**category** into dummies. As the factors are ordered alphabetically,
*Markham* is used as the base city and *Asian* as the base restaurant
type.

``` r
levels(yelp$city)
```

    ## [1] "Markham"    "Missisauga" "Toronto"

``` r
levels(yelp$categories)
```

    ## [1] "Asian"                "Coffee or Sandwiches" "Fast Food"           
    ## [4] "Italian"              "Latin"                "Middle Eastern"      
    ## [7] "North American"       "Other"                "Pub"

``` r
model1 <- glm(formula = class ~ ., data =  train, 
              family = "binomial")

summary(model1)
```

    ## 
    ## Call:
    ## glm(formula = class ~ ., family = "binomial", data = train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.7239  -0.9402  -0.7755   1.3115   1.9938  
    ## 
    ## Coefficients:
    ##                                  Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                    -1.6466697  0.1078541 -15.268  < 2e-16 ***
    ## cityMissisauga                  0.6681318  0.1225124   5.454 4.94e-08 ***
    ## cityToronto                     0.6171498  0.1065188   5.794 6.88e-09 ***
    ## review_count                    0.0025447  0.0003922   6.489 8.66e-11 ***
    ## categoriesCoffee or Sandwiches  0.7976169  0.0863991   9.232  < 2e-16 ***
    ## categoriesFast Food            -0.2012859  0.0871545  -2.310 0.020914 *  
    ## categoriesItalian               0.5134151  0.1525401   3.366 0.000763 ***
    ## categoriesLatin                 0.6149274  0.1241303   4.954 7.27e-07 ***
    ## categoriesMiddle Eastern        0.3759002  0.0873435   4.304 1.68e-05 ***
    ## categoriesNorth American        0.2514706  0.0952780   2.639 0.008307 ** 
    ## categoriesOther                 0.6181148  0.0964571   6.408 1.47e-10 ***
    ## categoriesPub                   0.5974255  0.1035668   5.769 8.00e-09 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 9454.6  on 7374  degrees of freedom
    ## Residual deviance: 9171.0  on 7363  degrees of freedom
    ## AIC: 9195
    ## 
    ## Number of Fisher Scoring iterations: 4

------------------------------------------------------------------------

##### 5. Predicting the probability of a **coffee shop** in **Toronto** to be ranked as above average (**class** = 1), expecting to receive 200 reviews.

``` r
# predict function
predict(model1, 
        newdata=data.frame(city = "Toronto", 
                           review_count = 200, 
                           categories = "Coffee or Sandwiches"), 
        type="response")
```

    ##         1 
    ## 0.5688206

``` r
# manual calculation
b0 <- model1$coefficients["(Intercept)"][[1]]
b1 <- model1$coefficients["cityToronto"][[1]]
b2 <- model1$coefficients["review_count"][[1]]
b3 <- model1$coefficients["categoriesCoffee or Sandwiches"][[1]]

a <- b0 + b1 + b2*200 + b3
exp(a)/(1+exp(a))
```

    ## [1] 0.5688206

------------------------------------------------------------------------

##### 6. Evaluating the performance of the model based on accuracy.

``` r
predictions <- predict(model1, 
                       newdata = test, 
                       type="response") 
predictions <- ifelse(predictions > 0.5, 1, 0)
confusionMatrix(data = as.factor(predictions), reference =  as.factor(test$class))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 1193  606
    ##          1   19   26
    ##                                           
    ##                Accuracy : 0.6611          
    ##                  95% CI : (0.6389, 0.6827)
    ##     No Information Rate : 0.6573          
    ##     P-Value [Acc > NIR] : 0.3758          
    ##                                           
    ##                   Kappa : 0.0327          
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##             Sensitivity : 0.98432         
    ##             Specificity : 0.04114         
    ##          Pos Pred Value : 0.66315         
    ##          Neg Pred Value : 0.57778         
    ##              Prevalence : 0.65727         
    ##          Detection Rate : 0.64696         
    ##    Detection Prevalence : 0.97560         
    ##       Balanced Accuracy : 0.51273         
    ##                                           
    ##        'Positive' Class : 0               
    ## 

*Based on CMTH642 - Advanced Data Analytics, Lab 7*
