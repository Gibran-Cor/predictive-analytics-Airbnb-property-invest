rm(list=ls())

library("data.table")
library("ggplot2")
library("GGally")
library(mice)
library(dplyr)
library(naniar)
library(readr)
library(glmnet)
library(rpart)
library(dplyr)
library(ggplot2)
library(caret)


library(glmnet)
airbnb = read.csv(file = "airbnb_updated_columns.csv")

#View(airbnb)

head(airbnb)

summary(airbnb)

# checking data types
str(airbnb)

# checking for null values
summary(is.na(airbnb))

# transformations
airbnb$price = parse_number(airbnb$price)
airbnb$security_deposit = parse_number(airbnb$security_deposit)
airbnb$extra_people = parse_number(airbnb$extra_people)

#airbnb$cancellation_policy[airbnb$cancellation_policy == "flexible"] <- 1
#airbnb$cancellation_policy[airbnb$cancellation_policy == "moderate"] <- 2
#airbnb$cancellation_policy[airbnb$cancellation_policy == "strict"] <- 3
#airbnb$cancellation_policy[airbnb$cancellation_policy == "strict_14_with_grace_period"] <- 4
#airbnb$cancellation_policy[airbnb$cancellation_policy == "super_strict_30"] <- 5
#airbnb$cancellation_policy[airbnb$cancellation_policy == "super_strict_60"] <- 6

str(airbnb)
glimpse(airbnb)
# Add additional variables
airbnb <- mutate(airbnb, 
                 log_price = log(price),
                 property_type_factor = as.factor(property_type),
                 neighbourhood_factor = as.factor(neighbourhood),
                 cancellation_policy_factor = as.factor(cancellation_policy),)

airbnb

airbnb$neighbourhood_encoded <- as.numeric(airbnb$neighbourhood_factor)
airbnb$cancellation_policy_encoded <- as.numeric(airbnb$cancellation_policy_factor)
airbnb_model <- lm(price ~ neighbourhood_encoded + accommodates + bathrooms + bedrooms + 
                     beds + amenities_count + guests_included + 
                     availability_365 + number_of_reviews + cancellation_policy_encoded,
                   data = airbnb)


summary(airbnb_model)

summary_model_1 <- summary(airbnb_model)
mse_1 <- summary_model_1$sigma^2; mse_1
r_sq_1 <- summary_model_1$r.squared; r_sq_1
adj_r_sq_1 <- summary_model_1$adj.r.squared; adj_r_sq_1

summary_model_1

par(mfrow=c(2,2)) 
plot(airbnb_model)




# Create interaction terms
airbnb$bedrooms_accommodates <- airbnb$bedrooms * airbnb$accommodates
airbnb$bathrooms_accommodates <- airbnb$bathrooms * airbnb$accommodates

airbnb$log_price[is.infinite(airbnb$log_price)] <- max(airbnb$log_price)
mean_val<-mean(airbnb$log_price,na.rm=TRUE)
airbnb$log_price[is.na(airbnb$log_price)] <- 1 
airbnb$log_price[is.nan(airbnb$log_price)] <- 1

log_model<-lm(log_price ~  neighbourhood_encoded + accommodates + bathrooms + bedrooms + 
                beds + amenities_count + guests_included + bedrooms_accommodates+bathrooms_accommodates+
                availability_365 + number_of_reviews + cancellation_policy, data = airbnb)
summary(log_model)



# Fit a nonlinear model
nonlinear_model <- nls(log_price ~ a + b * log(neighbourhood_encoded) + c * log(bedrooms + 1), 
                       data = airbnb, 
                       start = list(a = 100, b = 200, c = 500))
summary(nonlinear_model)
tree_model <- rpart(log_price ~ neighbourhood_encoded + accommodates + bathrooms + bedrooms + 
                      beds + amenities_count + guests_included + 
                      availability_365 + number_of_reviews + cancellation_policy, 
                    data = airbnb, 
                    method = "anova",
                    control = rpart.control(cp = 0.001))
summary(tree_model)


# Split data into training and testing sets
set.seed(123)
airbnb<-na.omit(airbnb)
train_index <- sample(1:nrow(airbnb), 0.6*nrow(airbnb))
train_data <- airbnb[train_index, ]
test_data <- airbnb[-train_index, ]

# Prepare the data
x <- model.matrix(log_price ~ neighbourhood_encoded + accommodates + bathrooms + bedrooms + 
                    beds + amenities_count + guests_included + bedrooms_accommodates+bathrooms_accommodates+
                    availability_365 + number_of_reviews + cancellation_policy, 
                  data = train_data)[,-1] # remove the intercept
y <- train_data$log_price

# Fit the lasso regression model
lasso_model <- glmnet(x, y, alpha = 1)
print(lasso_model)

# Fit the ridge regression model
ridge_model <- glmnet(x, y, alpha = 0)
print(ridge_model)

# Evaluate the models on the test set
x_test <- model.matrix(log_price ~ neighbourhood_encoded + accommodates + bathrooms + bedrooms + 
                         beds + amenities_count + guests_included + bedrooms_accommodates+bathrooms_accommodates+
                         availability_365 + number_of_reviews + cancellation_policy, 
                       data = test_data)[,-1] # remove the intercept
y_test <- test_data$log_price

# Predict using the lasso model
lasso_pred <- predict(lasso_model, newx = x_test)
lasso_rmse <- sqrt(mean((lasso_pred - y_test)^2))
print(paste("Lasso RMSE:", lasso_rmse))

# Predict using the ridge model
ridge_pred <- predict(ridge_model, newx = x_test)
ridge_rmse <- sqrt(mean((ridge_pred - y_test)^2))
print(paste("Ridge RMSE:", ridge_rmse))

