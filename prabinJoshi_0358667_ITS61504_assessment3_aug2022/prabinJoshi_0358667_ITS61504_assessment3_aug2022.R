# Load libraries
library(tidyverse)  # Data manipulation and visualization
library(caret)      # Data preprocessing and modeling
library(e1071)      # Support Vector Machines
library(class)      # k-Nearest Neighbors algorithm
library(ROSE)       # Package for handling class imbalance

# Dataset Basic Information
diabetes_data <- read.csv("diabetes.csv")

#for EDA part:
# Display basic information about the dataset
str(diabetes_data)

# Summary Statistics for Numerical Variables
summary_stats_numerical <- summary(diabetes_data[, c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age")])
print(summary_stats_numerical)

# Function to get a count and unique values summary for categorical variables including frequency percentage
categorical_summary <- function(column_name) {
  cat_summary <- table(diabetes_data[[column_name]])
  unique_values <- length(unique(diabetes_data[[column_name]]))
  total_count <- sum(cat_summary)
  percentage <- prop.table(cat_summary) * 100  # Calculate percentage
  cat_summary_df <- data.frame(
    Category = names(cat_summary),
    Count = as.numeric(cat_summary),
    Percentage = percentage
  )
  print(paste("Summary for", column_name, ":"))
  print(paste("Unique Values:", unique_values))
  print("Count summary:")
  print(cat_summary_df)
}

# Apply the updated function to the outcome variable
categorical_summary("Outcome")

# Correlation matrix
correlation_matrix <- cor(diabetes_data[, -10]) 
print("Correlation Matrix:")
print(correlation_matrix)

library(heatmaply)
# Plotting the correlation heatmap
heatmaply_cor(x = correlation_matrix, 
              xlab = "Features", 
              ylab = "Features", 
              k_col = 2,  # Number of clusters for columns
              k_row = 2   # Number of clusters for rows
)

# Univariate Analysis
# Continuous Variables - Histograms
continuous_columns <- c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age")
par(mfrow=c(3, 3))  # Set up a 3x3 grid for subplots
for (column_name in continuous_columns) {
  hist(diabetes_data[[column_name]], main=column_name, xlab=column_name, col="skyblue", border="black")
}

# Categorical Variables - Bar Plots
categorical_columns <- c("Outcome")
par(mfrow=c(1, 1))  # Set up a 1x1 grid for subplots
barplot(table(diabetes_data$Outcome), main="Outcome Distribution",
        col=c("skyblue", "lightcoral"), border="black",
        xlab="Outcome", ylab="Count", names.arg=c("No Diabetes", "Diabetes"))

# Scatterplot of Glucose vs. Age with Outcome Coloring
ggplot(diabetes_data, aes(x = Age, y = Glucose, color = factor(Outcome))) +
  geom_point() +
  labs(title = "Scatterplot of Glucose vs. Age with Outcomes", x = "Age", y = "Glucose Level", color = "Outcome") +
  scale_color_manual(values = c("skyblue", "lightgreen"))

# Data preprocessing part
# Check for missing values in the dataset
missing_values <- colSums(is.na(diabetes_data))
print("Missing Values Summary:")
print(missing_values)

# Check Class Imbalance
# Check class distribution
class_distribution <- table(diabetes_data$Outcome)
print("Class Distribution:")
print(class_distribution)

#Balancing the Class
# Perform class balancing using Synthetic Minority Over-sampling Technique (SMOTE) from ROSE
balanced_data <- ovun.sample(Outcome ~ ., data = diabetes_data, method = "over", N = 2 * sum(diabetes_data$Outcome == 0), seed = 42)$data

# Check class distribution after balancing
balanced_class_distribution <- table(balanced_data$Outcome)
print("Balanced Class Distribution:")
print(balanced_class_distribution)

# Boxplots to identify outliers in numerical variables
par(mfrow=c(3, 3))  # Set up a 3x3 grid for subplots
for (column_name in continuous_columns) {
  boxplot(diabetes_data[[column_name]], main=paste("Boxplot of", column_name), col="skyblue", border="black")
}

# Identify outliers in numerical variables
outlier_detection <- function(column) {
  Q1 <- quantile(diabetes_data[[column]], 0.25)
  Q3 <- quantile(diabetes_data[[column]], 0.75)
  IQR <- Q3 - Q1
  outliers <- diabetes_data[[column]] < (Q1 - 1.5 * IQR) | diabetes_data[[column]] > (Q3 + 1.5 * IQR)
  return(outliers)
}

# Identify outliers in numerical variables
outliers <- sapply(continuous_columns, outlier_detection)
print("Number of Outliers in Each Numerical Variable:")
print(colSums(outliers))

# Remove outliers from the dataset
diabetes_data_no_outliers <- diabetes_data[!apply(outliers, 1, any), ]

# Display information before and after removing outliers
cat("Original dataset size:", nrow(diabetes_data), "rows\n")
cat("Dataset size after removing outliers:", nrow(diabetes_data_no_outliers), "rows\n")

# Update diabetes_data to contain the dataset without outliers
diabetes_data <- diabetes_data_no_outliers

# Split the dataset into training and testing sets
set.seed(123)  # Setting seed for reproducibility
sample_indices <- sample(1:nrow(diabetes_data), 0.8 * nrow(diabetes_data))
train_data <- diabetes_data[sample_indices, ]
test_data <- diabetes_data[-sample_indices, ]

# Define predictor variables and target variable
predictors <- c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age")
target <- "Outcome"

# Feature scaling using the preProcess function from caret
scaling_model <- preProcess(train_data[, predictors], method = c("center", "scale"))

# Apply the scaling transformation to both the training and test sets
train_data_scaled <- predict(scaling_model, train_data[, predictors])
test_data_scaled <- predict(scaling_model, test_data[, predictors])

# Build the KNN model on the scaled data
knn_model_scaled <- knn(
  train = train_data_scaled,
  test = test_data_scaled,
  cl = train_data[, target],  # Use the original target variable
  k = 5
)

print("Confusion Matrix for Scaled KNN:")
# Evaluate the performance of the scaled KNN model
conf_matrix_knn_scaled <- table(knn_model_scaled, test_data$Outcome)
accuracy_knn_scaled <- sum(diag(conf_matrix_knn_scaled)) / sum(conf_matrix_knn_scaled)
precision_knn_scaled <- conf_matrix_knn_scaled[2, 2] / sum(conf_matrix_knn_scaled[, 2])
recall_knn_scaled <- conf_matrix_knn_scaled[2, 2] / sum(conf_matrix_knn_scaled[2, ])
f1_score_knn_scaled <- 2 * (precision_knn_scaled * recall_knn_scaled) / (precision_knn_scaled + recall_knn_scaled)

# Display the confusion matrix and performance metrics for scaled KNN
print("Confusion Matrix for Scaled KNN:")
print(conf_matrix_knn_scaled)
cat("\nAccuracy for Scaled KNN:", round(accuracy_knn_scaled, 3))
cat("\nPrecision for Scaled KNN:", round(precision_knn_scaled, 3))
cat("\nRecall for Scaled KNN:", round(recall_knn_scaled, 3))
cat("\nF1 Score for Scaled KNN:", round(f1_score_knn_scaled, 3))


# Build the SVM model for classification
svm_model <- svm(
  formula = as.formula(paste(target, "~", paste(predictors, collapse = "+"))),
  data = train_data,
  type = "C-classification",  # Change to C-classification
  kernel = "linear",
  cost = 1
)

# Make predictions on the test set
svm_predictions <- predict(svm_model, newdata = test_data)

# Evaluate the performance of the SVM model
conf_matrix_svm <- table(svm_predictions, test_data$Outcome)
accuracy_svm <- sum(diag(conf_matrix_svm)) / sum(conf_matrix_svm)
precision_svm <- conf_matrix_svm[2, 2] / sum(conf_matrix_svm[, 2])
recall_svm <- conf_matrix_svm[2, 2] / sum(conf_matrix_svm[2, ])
f1_score_svm <- 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm)

# Display the confusion matrix and performance metrics for SVM
print("Confusion Matrix for SVM:")
print(conf_matrix_svm)
cat("\nAccuracy for SVM:", round(accuracy_svm, 3))
cat("\nPrecision for SVM:", round(precision_svm, 3))
cat("\nRecall for SVM:", round(recall_svm, 3))
cat("\nF1 Score for SVM:", round(f1_score_svm, 3))

# Model Comparison
# Create a data frame to store the performance metrics
model_metrics <- data.frame(
  Model = c("KNN Scaled", "SVM"),
  Accuracy = numeric(2),
  Precision = numeric(2),
  Recall = numeric(2),
  F1_Score = numeric(2)
)

# Update the model_metrics data frame with KNN Scaled metrics
model_metrics[1, 2:5] <- c(accuracy_knn_scaled, precision_knn_scaled, recall_knn_scaled, f1_score_knn_scaled)

# Update the model_metrics data frame with SVM metrics
model_metrics[2, 2:5] <- c(accuracy_svm, precision_svm, recall_svm, f1_score_svm)

# Display the model comparison results
print("Model Comparison:")
print(model_metrics)

# Model Comparison Visualization
# Plotting the accuracy, precision, recall, and F1 Score for each model
par(mfrow=c(2, 2))

# Accuracy Comparison
barplot(model_metrics$Accuracy, names.arg = model_metrics$Model, main = "Accuracy Comparison", col = rainbow(2), ylim = c(0, 1))

# Precision Comparison
barplot(model_metrics$Precision, names.arg = model_metrics$Model, main = "Precision Comparison", col = rainbow(2), ylim = c(0, 1))

# Recall Comparison
barplot(model_metrics$Recall, names.arg = model_metrics$Model, main = "Recall Comparison", col = rainbow(2), ylim = c(0, 1))

# F1 Score Comparison
barplot(model_metrics$F1_Score, names.arg = model_metrics$Model, main = "F1 Score Comparison", col = rainbow(2), ylim = c(0, 1))


