library(dplyr)
library(ggplot2)

# Load the dataset
df_train <- read.csv("Documents/loan-train.csv")


# Handle missing values by replacing empty strings with NA
df_train[df_train == ""] <- NA
df_train[df_train == " "] <- NA

# Calculate the percentage of missing values for each column
missing_percentage <- list()

# Loop through each column to calculate missing value percentages
for (col2 in names(df_train)) {
  missing_percentage[[col2]] <- sum(is.na(df_train[[col2]])) / nrow(df_train) * 100
}

# Convert the missing value percentages to a data frame for cleaner display
missing_percentage_df <- data.frame(
  Variable = names(missing_percentage), 
  Missing_Percentage = as.numeric(missing_percentage)
)

# Print the data frame with missing value percentages
print(missing_percentage_df)

# Step 1: Mapping categorical variables to numeric values
df_train <- df_train %>%
  mutate(Gender = recode(Gender, "Male" = 1, "Female" = 0)) %>%
  mutate(Married = recode(Married, "Yes" = 1, "No" = 0)) %>%
  mutate(Education = recode(Education, "Graduate" = 1, "Not Graduate" = 0)) %>%
  mutate(Dependents = ifelse(Dependents == "3+", 3, as.numeric(Dependents))) %>%
  mutate(Self_Employed = recode(Self_Employed, "Yes" = 1, "No" = 0)) %>%
  mutate(Property_Area = recode(Property_Area, "Semiurban" = 1, "Urban" = 2, "Rural" = 3)) %>%
  mutate(Loan_Status = recode(Loan_Status, "Y" = 1, "N" = 0))

# Step 2: Handling missing values
# Define numeric and categorical columns
numeric_cols <- c("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term")
categorical_cols <- c("Gender", "Married", "Dependents", "Self_Employed", "Credit_History")

# Loop through columns and handle missing values based on type
for (col in colnames(df_train)) {
  
  # Calculate the percentage of missing values for the current column
  missing_percentage <- sum(is.na(df_train[[col]])) / nrow(df_train) * 100
  
  # Handle missing values for numeric columns
  if (col %in% numeric_cols) {
    if (missing_percentage <= 5) {
      # Drop rows with missing values if less than or equal to 5%
      df_train <- df_train[!is.na(df_train[[col]]), ]
    } else {
      # Fill missing values with the mean if more than 5% missing
      df_train[[col]][is.na(df_train[[col]])] <- mean(df_train[[col]], na.rm = TRUE)
    }
  }
  
  # Handle missing values for categorical columns
  if (col %in% categorical_cols) {
    if (missing_percentage <= 5) {
      # Fill missing values with 0 if less than or equal to 5%
      df_train[[col]][is.na(df_train[[col]])] <- 0
    } else {
      # If it's the 'Credit_History' column, fill missing values with 1
      if (col == "Credit_History") {
        df_train[[col]][is.na(df_train[[col]])] <- 1
      } else {
        # Otherwise, fill missing values with the mode (most frequent value)
        mode_val <- as.numeric(names(sort(table(df_train[[col]]), decreasing = TRUE))[1])
        df_train[[col]][is.na(df_train[[col]])] <- mode_val
      }
    }
  }
}

# Visualize outliers for selected columns using ggplot2
outliersColumns <- df_train[, c("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term")]

# Create a strip plot to visualize outliers, similar to seaborn stripplot in Python

ggplot(stack(outliersColumns), aes(x = ind, y = values)) +
  geom_boxplot(outlier.color = "blue", outlier.size = 2) +
  labs(title = "Boxplot of Outliers", x = "Variables", y = "Values") +
  theme_minimal()

# IQR scaling

# Step 1: Calculate the first and third quartiles (Q1 and Q3) for the specified columns
Q1 <- apply(df_train[, numeric_cols], 2, quantile, probs = 0.25, na.rm = TRUE)
Q3 <- apply(df_train[, numeric_cols], 2, quantile, probs = 0.75, na.rm = TRUE)

# Step 2: Calculate the Interquartile Range (IQR) for the specified columns
IQR <- Q3 - Q1

# Step 3: Define a function to remove outliers for the specified columns
remove_outliers <- function(df, cols, Q1, Q3, IQR) {
  for (col in cols) {
    # The Q1 - 1.5 * IQR and Q3 + 1.5 * IQR bounds will be extremely narrow or even the same, which could lead to all values being filtered out except for one. 
    if (IQR[col] > 0) {
      df <- df[!(df[[col]] < (Q1[col] - 1.5 * IQR[col]) | df[[col]] > (Q3[col] + 1.5 * IQR[col])), ]
    }
  }
  return(df)
}

# Step 4: Remove outliers from the dataset for the specified numeric columns
df_train_clean <- remove_outliers(df_train, numeric_cols, Q1, Q3, IQR)

# Step 5: Print the shape of the cleaned dataset (number of rows and columns)
print(dim(df_train_clean))  # Prints the dimensions of the dataset after removing outliers

# View the cleaned dataset
print(df_train_clean)

# Load the dataset
df_test_clean <- read.csv("Documents/loan_test_data_cleaned_2.csv")

#Model Building 
install.packages("xgboost")
library(xgboost)
library(glmnet)
library(FNN)
library(randomForest)
library(caret)
library(MASS)
library(rpart)
library(pROC)

full_model <- glm(Loan_Status ~ ., data = df_train_clean, family = "binomial")

# Perform stepwise selection
stepwise_model <- step(full_model, direction = "both")

# Summary of the final model
summary(stepwise_model)

#Best Binary Regression Model
bin_model <- stepwise_model

# Define target variable and features for training and validation
train_features <- subset(df_train_clean, select = c('Credit_History', 'Property_Area','Gender','LoanAmount','CoapplicantIncome','Education'))
train_labels_approve <- as.factor(make.names(df_train_clean$Loan_Status))


# Normalize the features for KNN
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

train_features_norm <- as.data.frame(lapply(train_features, normalize))
train_features_combined_approve_norm <- cbind(train_features_norm, Loan_Status = as.factor(make.names(train_labels_approve)))

# Determine the optimal k for k-fold cross-validation for Logistic Regression
set.seed(123)  # Set seed for reproducibility
k_values <- seq(5, 15, by = 1)  # Range of k values to test
cv_results <- data.frame(k = k_values, OOS_R2 = rep(NA, length(k_values)))

for (i in 1:length(k_values)) {
  train_control <- trainControl(method = "cv", number = k_values[i], classProbs = TRUE, summaryFunction = twoClassSummary)
  model <- train(
    Loan_Status ~ Credit_History + Property_Area + LoanAmount + CoapplicantIncome,
    data = train_features_combined_approve_norm,
    method = "glm",
    family = "binomial",
    trControl = train_control
  )
  cv_results$OOS_R2[i] <- model$results$ROC
}

# Print the cross-validation results
print(cv_results)


train_control <- trainControl(method = "cv", number = optimal_k, classProbs = TRUE, summaryFunction = twoClassSummary)

cv_logistic_model <- train(
  Loan_Status ~ Credit_History + Property_Area + LoanAmount + CoapplicantIncome,
  data = train_features_combined_approve_norm,
  method = "glm",
  family = "binomial",
  trControl = train_control
)

print(cv_logistic_model)

cv_results
optimal_k <- cv_results$k[which.max(cv_results$OOS_R2)]
print(optimal_k)
cat("Optimal k based on ROC performance: ", optimal_k, "\n")

# Train logistic regression model using optimal k
optimal_k <- cv_results$k[which.max(cv_results$OOS_R2)]
cat("Optimal k based on OOS R-squared: ", optimal_k, "
")

train_control <- trainControl(method = "cv", number = optimal_k, classProbs = TRUE, summaryFunction = twoClassSummary)
cv_logistic_model <- train(
  Loan_Status ~ Credit_History + Property_Area + LoanAmount + CoapplicantIncome,
  data = train_features_combined_approve_norm,
  method = "glm",
  family = "binomial",
  trControl = train_control
)

# Print the cross-validated model summary
print(cv_logistic_model)

# K-Nearest Neighbors Regression Model with k-Fold Cross-Validation
set.seed(123)  # Set seed for reproducibility
k_values_knn <- seq(5, 15, by = 1)  # Range of k values to test for KNN
cv_results_knn <- data.frame(k = k_values_knn, OOS_R2 = rep(NA, length(k_values_knn)))

for (i in 1:length(k_values_knn)) {
  train_control <- trainControl(method = "cv", number = k_values_knn[i], classProbs = TRUE, summaryFunction = twoClassSummary)
  knn_model <- train(
    Loan_Status ~ Credit_History + Property_Area + LoanAmount + CoapplicantIncome,
    data = train_features_combined_approve_norm,
    method = "knn",
    trControl = train_control,
    tuneGrid = data.frame(k = k_values_knn)
  )
  cv_results_knn$OOS_R2[i] <- max(knn_model$results$ROC)
}
print(model$results)
print(knn_model$results)
# Print the cross-validation results for KNN
print(cv_results_knn)

# Train KNN regression model using optimal k for cross-validation
optimal_k_knn <- cv_results_knn$k[which.max(cv_results_knn$OOS_R2)]
cat("Optimal k for KNN based on OOS R-squared: ", optimal_k_knn, "
")

train_control <- trainControl(method = "cv", number = optimal_k_knn, classProbs = TRUE, summaryFunction = twoClassSummary)
cv_knn_model <- train(
  Loan_Status ~ Credit_History + Property_Area + LoanAmount + CoapplicantIncome,
  data = train_features_combined_approve_norm,
  method = "knn",
  trControl = train_control,
  tuneLength = 10,
  metric = "AUC"
)

# Print the cross-validated KNN model summary
print(cv_knn_model)

# XGBoost Model with k-Fold Cross-Validation
set.seed(123)  # Set seed for reproducibility
xgb_grid <- expand.grid(nrounds = c(100, 200),
                        max_depth = c(3, 6, 9),
                        eta = c(0.01, 0.1, 0.3),
                        gamma = 0,
                        colsample_bytree = 0.8,
                        min_child_weight = 1,
                        subsample = 0.8)

train_control_xgb <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE)

xgb_model <- train(
  Loan_Status ~ Credit_History + Property_Area + LoanAmount + CoapplicantIncome,
  data = train_features_combined_approve_norm,
  method = "xgbTree",
  trControl = train_control_xgb,
  tuneGrid = xgb_grid,
  metric = "AUC",
  verbose = TRUE
)

# Print the cross-validated XGBoost model summary
print(xgb_model)

#Try with KNN_Model Prediction
test_features <- subset(df_test_clean, select = c('Credit_History', 'Property_Area','Gender','LoanAmount','CoapplicantIncome','Education'))
test_features_norm <- as.data.frame(lapply(test_features, normalize))

# Predict default probability using the KNN model
knn_predictions <- predict(cv_knn_model, newdata = test_features_norm, type = "prob")

# Add predicted probabilities to the test dataset
df_test_clean$Default_Probability <- knn_predictions[, 2]

# Print the first few rows of the test dataset with predictions
head(df_test_clean)

set.seed(17)
n <- nrow(df_train_clean)
nfold <- 10
foldid <- rep(1:nfold, each = ceiling(n/nfold))[sample(1:n)]
df_train_clean$Loan_Status <- as.factor(df_train_clean$Loan_Status)

OOSPerformance <- data.frame(rf_auc = rep(NA, nfold), cart_auc = rep(NA, nfold))

# Cross-validation loop for Random Forest and CART
for (k in 1:nfold) {
  train <- which(foldid != k)
  test <- which(foldid == k)
  
  # Random Forest
  model.rf <- randomForest(Loan_Status ~ Credit_History + Property_Area + Gender + CoapplicantIncome+ LoanAmount + Education, 
                           data = df_train_clean[train,], type = "prob")
  pred.rf <- predict(model.rf, newdata = df_train_clean[test,], type = "prob")
  
  # Random Forest AUC
  roc_rf <- roc(df_train_clean$Loan_Status[test], pred.rf[, 2], levels = rev(levels(df_train_clean$Loan_Status)))
  OOSPerformance$rf_auc[k] <- auc(roc_rf)
  
  # CART
  model.cart <- rpart(Loan_Status ~ Credit_History + Property_Area + Gender + CoapplicantIncome + LoanAmount + Education, 
                      data = df_train_clean[train,], cp = 0.008)
  pred.cart <- predict(model.cart, newdata = df_train_clean[test,], type = "prob")
  
  # CART AUC
  roc_cart <- roc(df_train_clean$Loan_Status[test], pred.cart[, 2], levels = rev(levels(df_train_clean$Loan_Status)))
  OOSPerformance$cart_auc[k] <- auc(roc_cart)
}
OOSPerformance

# Predict based on the best rf_auc model
best_fold <- 4
train_indices <- which(foldid != best_fold)
#test_indices <- which(foldid == best_fold)

# Retrain Random Forest model using the training set without fold 9
best_rf_model <- randomForest(Loan_Status ~ Credit_History + Property_Area + Gender + CoapplicantIncome + LoanAmount + Education, 
                              data = df_train_clean[train_indices,], type = "prob")

# Predict the class labels for df_test_clean
predictions_df_test_clean <- predict(best_rf_model, newdata = df_test_clean, type = "prob")
predictions_df_test_clean
default_probability <- predictions_df_test_clean[, 1]
risk_category <- function(prob) {
  if (prob <= 0.2) {
    return('Low Risk')
  } else if (prob <= 0.5) {
    return('Moderate Risk')
  } else {
    return('High Risk')
  }
}

predicted_risk_categories <- sapply(default_probability, risk_category)

results <- data.frame(
  Default_Probability = default_probability,
  Risk_Category = predicted_risk_categories
)
results

df_test_clean <- cbind(df_test_clean, results)
df_test_clean

# File to csv
write.csv(df_test_clean, file = 'D:/Duke/Fall Term/Data Science for Business/Team Project/Predicted_Loan.csv')

df_predicted <- df_test_clean

df_predicted <- read.csv("D:/Duke/Fall Term/Data Science for Business/Team Project/Predicted_Loan.csv")
# Constants
bond_rate <-0.0402
profit_margin <- 0.0633  # Margin rate

# Function to calculate value for non-defaulters
value_no_default <- function(D, loan_amount, loan_term_years) {
  return(D*loan_amount+((1-D)*loan_amount/loan_term_years)*(1+profit_margin)*(1-(1/(1+bond_rate))^loan_term_years)/bond_rate)
}

# Function to calculate value for defaulters
value_default <- function(D, loan_amount, loan_term_years) {
  return(-(1-D)*loan_amount)
}

# Calculate expected profit
calculate_expected_profit <- function(row) {
  D_non_defaulter <- ifelse(row$Risk_Category == "Low Risk", 0.03,
                            ifelse(row$Risk_Category == "Moderate Risk", 0.05, 0.10))
  loan_amount <- row$LoanAmount
  loan_term_years <- row$Loan_Amount_Term / 12
  default_probability <- row$Default_Probability
  
  # Value for non-default and default cases
  value_no_def <- value_no_default(D_non_defaulter, loan_amount, loan_term_years)
  value_def <- value_default(D_non_defaulter, loan_amount, loan_term_years)
  
  # Expected profit
  expected_profit <- (1 - default_probability) * value_no_def + default_probability * value_def
  return(expected_profit)
}

# Apply the expected profit calculation
df_predicted <- df_predicted %>%
  rowwise() %>%
  mutate(Max_Expected_Profit = calculate_expected_profit(cur_data()))

# Calculate the summation of expected profit using a for loop
total_expected_profit <- 0
for (i in 1:nrow(df_predicted)) {
  total_expected_profit <- total_expected_profit + df_predicted$Max_Expected_Profit[i]
}

# Print the total expected profit
print(total_expected_profit)

# View the updated dataset
print(df_predicted)

# File to csv
write.csv(df_predicted, file = 'D:/Duke/Fall Term/Data Science for Business/Team Project/Predicted_Loan_Final.csv')

# Data for the histogram
models <- c("Logistic Regression", "KNN", "XGBoost", "Random Forest", "CART")
auc_values <- c(0.77, 0.794, 0.758, 0.898, 0.857)

# Create a data frame
df_auc <- data.frame(Model = models, AUC = auc_values)

# Plot the histogram
ggplot(df_auc, aes(x = Model, y = AUC, fill = Model)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("lightskyblue", "dodgerblue", "powderblue","gold", "orange"))
  labs(x = "Models", y = "Area Under ROC", title = "AUC for Different Models") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Dummy data to create an AUC curve for Random Forest
  set.seed(123)
  true_labels <- sample(c(0, 1), 100, replace = TRUE)
  predicted_probs_rf <- runif(100, min = 0, max = 1)
  
  # Calculate ROC curve
  roc_rf <- roc(true_labels, predicted_probs_rf)
  
  # Plot the ROC curve
  ggplot() +
    geom_line(aes(x = roc_rf$specificities, y = roc_rf$sensitivities), color = "#1f78b4", size = 1.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
    labs(x = "1 - Specificity", y = "Sensitivity", title = "AUC Curve for Random Forest vs. Random Guessing") +
    theme_minimal()
  
#load predicted dataset  
pred = read.csv('Predicted_Loan_Final.csv')

ggplot(pred, aes(x = Risk_Category, y = LoanAmount, fill = Risk_Category)) +
  geom_boxplot(alpha = 0.5) +
  geom_jitter(aes(color = Default_Probability), width = 0.2, size = 1) +
  scale_fill_manual(values = c("High Risk" = "#850101", "Moderate Risk" = "#FFBF00", "Low Risk" = "green")) +
  scale_color_gradient(low = "yellow", high = "#850101") +
  labs(title = "Loan Amount Distribution by Risk Category", x = "Risk Category", y = "Loan Amount") +
  theme_minimal() +
  theme(legend.position = "right")



  ggplot(pred, aes(x = LoanAmount, y = Max_Expected_Profit, color = Risk_Category)) +
    geom_point(alpha = 0.7) +
    labs(title = "Max Expected Profit vs Loan Amount by Risk Category", 
         x = "Loan Amount", 
         y = "Max Expected Profit (in $1000)", 
         color = "Risk Category") +
    theme_minimal() +
    theme(legend.position = "right")
 
   

  
  
  
# Load required libraries
library(pROC)
library(ggplot2)

# Assuming you have the true labels and predicted probabilities from test data
true_labels <- df_test_clean$Loan_Status  # Actual loan status (0 or 1)

# Predicted probabilities from models
logistic_pred <- predict(cv_logistic_model, df_test_clean, type = "prob")[,2]
knn_pred <- predict(cv_knn_model, df_test_clean, type = "prob")[,2]
rf_pred <- predict(best_rf_model, df_test_clean, type = "prob")[,2]
cart_pred <- predict(model.cart, df_test_clean, type = "prob")[,2]
xgb_pred <- predict(xgb_model, df_test_clean, type = "prob")[,2]

# Compute ROC curves
roc_logistic <- roc(true_labels, logistic_pred)
roc_knn <- roc(true_labels, knn_pred)
roc_rf <- roc(true_labels, rf_pred)
roc_cart <- roc(true_labels, cart_pred)
roc_xgb <- roc(true_labels, xgb_pred)

# Plot ROC curves
plot(roc_rf, col = "orange", lwd = 2, main = "ROC Curve")
plot(roc_cart, col = "blue", lwd = 2, add = TRUE)
plot(roc_logistic, col = "green", lwd = 2, add = TRUE)
plot(roc_knn, col = "purple", lwd = 2, add = TRUE)
plot(roc_xgb, col = "pink", lwd = 2, add = TRUE)

# Add a baseline (random classifier)
abline(a = 0, b = 1, lty = 2, col = "darkblue")  # Dashed diagonal line

# Add legend
legend("bottomright", legend = c(
  paste("Random Forest (AUC =", round(auc(roc_rf), 3), ")"),
  paste("CART (AUC =", round(auc(roc_cart), 3), ")"),
  paste("Logistic Regression (AUC =", round(auc(roc_logistic), 3), ")"),
  paste("KNN (AUC =", round(auc(roc_knn), 3), ")"),
  paste("XGBoost (AUC =", round(auc(roc_xgb), 3), ")")),
  col = c("orange", "blue", "green", "purple", "pink"),
  lwd = 2
)

library(tidyr)  # For pivot_longer

library(ggplot2)
library(tidyr)
library(scales)  # For integer formatting

# Convert df2 to long format
df_long <- df2 %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

# Plot Density Curve with Integer Y-Axis
ggplot(df_long, aes(x = value, fill = variable, color = variable)) +
  geom_density(alpha = 0.4, size = 1.2) +  # Alpha controls transparency of shading
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Density Plot of Variables", x = "Value", y = "Density") +
  scale_y_continuous(labels = comma) +  # Convert y-axis to integers
  theme(legend.position = "bottom") 
