#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[211]:


# Basic Libraries
import pandas as pd
import numpy as np

# Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Model Persistence
import joblib

# Data Preprocessing
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

# Model Selection and Evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    f1_score
)

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Optimization Libraries
import optuna

# External Dataset Downloading
import opendatasets as od

# Progress Bar
from tqdm import tqdm as notebook_tqdm

# Set Plotting Defaults
sns.set(rc={'figure.figsize': (16, 8)})
sns.set_style("darkgrid")


# In[212]:


import os

def list_folders_and_files(directory):
    """
    List all folders, subfolders, and files in the specified directory without duplication.
    
    Args:
        directory (str): The root directory to start listing.
    """
    seen_directories = set()  # To keep track of visited directories
    for root, dirs, files in os.walk(directory):
        # Avoid duplicate directories
        if root in seen_directories:
            continue
        seen_directories.add(root)

        # Calculate the indentation level for the folder hierarchy
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")  # Print folder name
        
        # Print files in the current directory
        sub_indent = ' ' * 4 * (level + 1)
        for f in sorted(files):  # Sort files alphabetically
            print(f"{sub_indent}{f}")
        # Sort directories alphabetically to avoid potential reordering
        dirs.sort()

# Specify the root directory of your project
project_directory = input("Enter the path to your project directory: ")
list_folders_and_files(project_directory)


# ## Data Loading and Preprocessing

# In[16]:


# Define the desired path for the raw data folder
raw_data_folder = r"C:\Users\User\ml-zoomcamp\capstone_1\data\raw"

# Ensure the raw data folder exists
os.makedirs(raw_data_folder, exist_ok=True)

# Download the dataset into the raw data folder
dataset_url = 'https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment/data'
od.download(dataset_url, data_dir=raw_data_folder)

print(f"Dataset downloaded and stored in: {raw_data_folder}")


# In[161]:


# Load the dataset
df = pd.read_csv(r"C:\Users\User\ml-zoomcamp\capstone_1\data\raw\air-quality-and-pollution-assessment\updated_pollution_dataset.csv")
df.head()


# In[162]:


# Check the data types of each column
df.info()


# In[163]:


# Check for missing values
df.isnull().sum()


# No missing values.

# In[164]:


# Transform column names: make lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(" ", "_")
df.columns


# ## Exploratory Data Analysis (EDA)

# In[165]:


# Check basic information about registration_data_test dataset
print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print('='*40)
print ("Features : \n" ,df.columns.tolist())
print('='*40)
print ("Unique values :  \n",df.nunique())


# In[166]:


# Check summary statistics
df.describe().T


# In[167]:


# Plot distributions for numerical features
def plot_distributions(data, features):
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


# In[168]:


# Select numerical features for visualization
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
plot_distributions(df, numerical_features)


# In[169]:


# Bar plot for categorical features
def plot_categorical_distribution(data, feature):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=feature, data=data) 
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.show()


# In[170]:


# Plot categorical features
categorical_features = df.select_dtypes(include=[object]).columns.tolist()
for feature in categorical_features:
    plot_categorical_distribution(df, feature)


# This is our target variable.

# In[171]:


# Check the distribution of the target variable
air_quality_count = df['air_quality'].value_counts()
air_quality_count


# In[172]:


# Proportions of air quality levels
air_quality_proportions = df['air_quality'].value_counts(normalize=True)
air_quality_proportions.plot.pie(autopct="%.1f%%", figsize=(6, 6), title="Air Quality Distribution")
plt.ylabel("")
plt.show()


# In[173]:


# Plot boxplots for numerical features to identify outliers
def plot_boxplots(data, features):
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=data[feature])
        plt.title(f"Boxplot of {feature}")
        plt.xlabel(feature)
        plt.show()


# In[174]:


# Plot the boxplots
plot_boxplots(df, numerical_features)


# In[175]:


# Selecting only numerical columns
numerical_columns = df.select_dtypes(include=[np.number])


# In[176]:


# Calculating the Interquartile Range (IQR) for each numerical column
Q1 = numerical_columns.quantile(0.25)
Q3 = numerical_columns.quantile(0.75)
IQR = Q3 - Q1

# Defining the lower and upper bounds for outlier detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Finding outliers by checking if values fall outside the bounds
outliers = (numerical_columns < lower_bound) | (numerical_columns > upper_bound)

# Printing outliers for each numerical column
for column in outliers.columns:
    print(f"Outliers in {column}:")
    print(df[outliers[column]][column])
    print()


# In[177]:


# Calculating the number of outliers in each numerical column
num_outliers = outliers.sum()

# Computing the percentage of outliers in each numerical column
total_rows = df.shape[0]
percentage_outliers = (num_outliers / total_rows) * 100

# Creating a DataFrame to store the results
outlier_stats = pd.DataFrame({
    'Num_Outliers': num_outliers,
    'Percentage_Outliers': percentage_outliers
})

# Displaying the outlier statistics
print("Outlier Statistics:")
print(outlier_stats)


# In[178]:


# Function to visualize outliers
def visualize_outliers(data, column, lower_bound, upper_bound):
    plt.figure(figsize=(12, 6))

    # Scatter plot to show outliers
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=data.index, y=data[column], color="blue")
    plt.axhline(lower_bound[column], color="red", linestyle="--", label="Lower Bound")
    plt.axhline(upper_bound[column], color="green", linestyle="--", label="Upper Bound")
    plt.title(f"Scatter Plot of {column} (with Outliers)")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.legend()

    # Boxplot to visualize outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[column], color="lightblue")
    plt.title(f"Boxplot of {column}")

    plt.tight_layout()
    plt.show()


# In[179]:


# Visualize outliers for affected columns
for col in ['pm2.5', 'pm10']:
    visualize_outliers(df, col, lower_bound, upper_bound)


# In[180]:


# Boxplots to compare numerical features across air quality levels
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="air_quality", y=feature)
    plt.title(f"{feature} Distribution Across Air Quality Levels")
    plt.show()


# In[181]:


# Select pollutant features
pollutant_features = ['pm2.5', 'pm10', 'no2', 'so2', 'co']

# Calculate the average of pollutants by air quality
avg_pollutants = df.groupby('air_quality')[pollutant_features].mean()

# Plotting the average pollutant levels
avg_pollutants.plot(kind='bar', figsize=(10, 6))
plt.title("Average Pollutant Levels by Air Quality Category")
plt.xlabel("Air Quality Category")
plt.ylabel("Average Concentration")
plt.legend(title="Pollutants")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# In[182]:


# Pairwise scatter plot with KDEs on the diagonal
sns.pairplot(df, hue="air_quality", diag_kind="kde", corner=True)
plt.show()


# In[183]:


# Correlation Analysis
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(16, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[184]:


# Identify key correlated pairs (threshold > 0.8 for strong correlation)
correlation_threshold = 0.8
key_correlated_pairs = []

# Iterate over the correlation matrix and find pairs above the threshold
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            feature_1 = correlation_matrix.index[i]
            feature_2 = correlation_matrix.columns[j]
            correlation_value = correlation_matrix.iloc[i, j]
            key_correlated_pairs.append((feature_1, feature_2, correlation_value))

# Convert to DataFrame for better readability and display
key_correlated_pairs_df = pd.DataFrame(key_correlated_pairs, columns=["Feature 1", "Feature 2", "Correlation"])
key_correlated_pairs_df = key_correlated_pairs_df.sort_values(by="Correlation", ascending=False).reset_index(drop=True)

key_correlated_pairs_df


# In[185]:


# KDE Plot
plt.figure(figsize=(8, 6))
sns.kdeplot(df['pm2.5'], fill=True, label='PM2.5', color='blue')
sns.kdeplot(df['pm10'], fill=True, label='PM10', color='orange')
plt.title("Distribution of PM2.5 and PM10")
plt.xlabel("Concentration (µg/m³)")
plt.ylabel("Density")
plt.legend()
plt.show()


# In[186]:


def correlation_ratio(categories, values):
    """
    Calculate the correlation ratio (eta-squared) for a categorical target and numerical features.

    Args:
        categories (pd.Series or array-like): Categorical target variable.
        values (pd.Series or array-like): Numerical feature variable.
    
    Returns:
        float: Correlation ratio value.
    """
    categories = np.array(categories)
    values = np.array(values)
    overall_mean = np.mean(values)
    category_means = [np.mean(values[categories == category]) for category in np.unique(categories)]
    category_sizes = [np.sum(categories == category) for category in np.unique(categories)]

    between_group_variance = sum(size * (mean - overall_mean) ** 2 for size, mean in zip(category_sizes, category_means))
    total_variance = np.sum((values - overall_mean) ** 2)
    
    return between_group_variance / total_variance if total_variance > 0 else 0


# In[187]:


numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
correlation_ratios = {}

for feature in numerical_features:
    correlation_ratios[feature] = correlation_ratio(df['air_quality'], df[feature])

# Convert to DataFrame for Sorting and Visualization
correlation_ratios_df = pd.DataFrame.from_dict(correlation_ratios, orient='index', columns=['Correlation Ratio'])
correlation_ratios_sorted = correlation_ratios_df.sort_values(by='Correlation Ratio', ascending=True)

# Plot the Correlation Ratios
correlation_ratios_sorted.plot(kind='barh', figsize=(12, 8))
plt.title("Correlation Ratios of Numerical Features with Target 'air_quality'")
plt.xlabel("Correlation Ratio")
plt.ylabel("Features")
plt.show()


# In[ ]:


# Normalize the pollutant features
scaler = MinMaxScaler()
normalized_pollutants = pd.DataFrame(
    scaler.fit_transform(df[pollutant_features]),
    columns=pollutant_features,
    index=df.index
)

# Calculate the mean of normalized pollutants by air quality
avg_normalized_pollutants = (
    normalized_pollutants.join(df['air_quality'])
    .groupby('air_quality')
    .mean()
)

# Prepare data for radar chart
categories = avg_normalized_pollutants.columns
num_vars = len(categories)

# Add a row to close the circle for radar chart
avg_normalized_pollutants = pd.concat([avg_normalized_pollutants, avg_normalized_pollutants.iloc[[0]]])

# Create radar chart
fig, ax = plt.subplots(figsize=(16, 8), subplot_kw=dict(polar=True))

# Create radar chart for each air quality category
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the circle

for air_quality, row in avg_normalized_pollutants.iterrows():
    values = row.tolist()
    values += values[:1]  # Close the circle
    ax.plot(angles, values, label=air_quality)
    ax.fill(angles, values, alpha=0.25)

# Customize the chart
ax.set_title("Normalized Radar Chart: Pollutants by Air Quality", size=16)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()


# In[189]:


# Dropping the 'pm2.5' feature
df = df.drop(columns=['pm2.5'])


# In[190]:


# Check the column names
df.columns


# ## Feature Engineering

# In[191]:


# Label encoding the target variable 'air_quality'
le = LabelEncoder()
df['air_quality_encoded'] = le.fit_transform(df['air_quality'])

# Define feature set (X) and target variable (y)
X = df.drop(columns=['air_quality', 'air_quality_encoded'])
y = df['air_quality_encoded']


# In[205]:


# Save the LabelEncoder
label_encoder_path = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed\label_encoder.pkl"
joblib.dump(le, label_encoder_path)

print(f"LabelEncoder saved at: {label_encoder_path}")


# In[192]:


# Split into train+validation and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split train+validation into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Identify numerical features for scaling
numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns

# Fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])

# Apply the scaler to validation and test sets
X_val_scaled = X_val.copy()
X_val_scaled[numerical_features] = scaler.transform(X_val[numerical_features])

X_test_scaled = X_test.copy()
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])


# In[193]:


# Save the scaler for later use
scaler_path = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed\scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved at: {scaler_path}")

# Check the scaled training data
print("\nShape of scaled training data:", X_train_scaled.shape)

# Check the scaled validation data
print("Shape of scaled validation data:", X_val_scaled.shape)

# Check the scaled test data
print("Shape of scaled test data:", X_test_scaled.shape)


# In[194]:


# Define the desired path for the processed data folder
processed_data_folder = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed"

# Ensure the processed data folder exists
os.makedirs(processed_data_folder, exist_ok=True)

# Save the datasets as CSV files
X_train_scaled.to_csv(os.path.join(processed_data_folder, "X_train_scaled.csv"), index=False)
y_train.to_csv(os.path.join(processed_data_folder, "y_train.csv"), index=False)

X_val_scaled.to_csv(os.path.join(processed_data_folder, "X_val_scaled.csv"), index=False)
y_val.to_csv(os.path.join(processed_data_folder, "y_val.csv"), index=False)

X_test_scaled.to_csv(os.path.join(processed_data_folder, "X_test_scaled.csv"), index=False)
y_test.to_csv(os.path.join(processed_data_folder, "y_test.csv"), index=False)

print("Processed datasets saved successfully in:", processed_data_folder)


# ## Modeling

# In[ ]:


# Define a dictionary of models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(force_col_wise=True),
    "CatBoost": CatBoostClassifier(verbose=0),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predict on validation set
    y_val_pred = model.predict(X_val_scaled)
    
    # Compute metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    weighted_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    # Store metrics
    results[name] = {
        "Accuracy": accuracy,
        "Weighted F1-Score": weighted_f1
    }
    
    # Display metrics
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("-" * 50)


# In[196]:


# Prepare the results for display in a table format
summary_results = pd.DataFrame(results).T
summary_results = summary_results.reset_index()
summary_results.columns = ["Model", "Accuracy", "Weighted F1-Score"]
summary_results = summary_results.sort_values(by="Weighted F1-Score", ascending=False)
summary_results


# ## Hyperparameter Tuning

# In[197]:


def objective_catboost(trial):
    # Define the hyperparameter space
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
    }

    # Create the CatBoost model
    model = CatBoostClassifier(verbose=0, random_state=42, **params)
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_val_pred = model.predict(X_val_scaled)
    weighted_f1 = f1_score(y_val, y_val_pred, average='weighted')

    return weighted_f1


# In[198]:


def objective_rf(trial):
    # Define the hyperparameter space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

    # Create the Random Forest model
    model = RandomForestClassifier(random_state=42, **params)
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_val_pred = model.predict(X_val_scaled)
    weighted_f1 = f1_score(y_val, y_val_pred, average='weighted')

    return weighted_f1


# In[199]:


def objective_xgb(trial):
    # Define the hyperparameter space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    # Create the XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, **params)
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_val_pred = model.predict(X_val_scaled)
    weighted_f1 = f1_score(y_val, y_val_pred, average='weighted')

    return weighted_f1


# In[200]:


def objective_lgbm(trial):
    # Define the hyperparameter space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", -1, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
    }

    # Create the LightGBM model
    model = LGBMClassifier(random_state=42, **params)
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_val_pred = model.predict(X_val_scaled)
    weighted_f1 = f1_score(y_val, y_val_pred, average='weighted')

    return weighted_f1


# In[201]:


# Run Optuna for each model
study_catboost = optuna.create_study(direction="maximize")
study_catboost.optimize(objective_catboost, n_trials=50)

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=50)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=50)

study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=50)

# Display best parameters and weighted F1-score for each model
print("Best CatBoost:", study_catboost.best_params, "Weighted F1-Score:", study_catboost.best_value)
print("Best Random Forest:", study_rf.best_params, "Weighted F1-Score:", study_rf.best_value)
print("Best XGBoost:", study_xgb.best_params, "Weighted F1-Score:", study_xgb.best_value)
print("Best LightGBM:", study_lgbm.best_params, "Weighted F1-Score:", study_lgbm.best_value)


# In[202]:


# Prepare a table summarizing the best results from Optuna
tuning_results = {
    "Model": ["CatBoost", "Random Forest", "XGBoost", "LightGBM"],
    "Best Weighted F1-Score": [
        study_catboost.best_value,
        study_rf.best_value,
        study_xgb.best_value,
        study_lgbm.best_value,
    ],
    "Best Parameters": [
        study_catboost.best_params,
        study_rf.best_params,
        study_xgb.best_params,
        study_lgbm.best_params,
    ]
}

# Convert to DataFrame
tuning_results_df = pd.DataFrame(tuning_results)
tuning_results_df = tuning_results_df.sort_values(by="Best Weighted F1-Score", ascending=False)
tuning_results_df


# In[203]:


# Identify the best model
best_model_index = tuning_results_df["Best Weighted F1-Score"].idxmax()
best_model_name = tuning_results_df.loc[best_model_index, "Model"]
best_model_params = tuning_results_df.loc[best_model_index, "Best Parameters"]
best_model_f1 = tuning_results_df.loc[best_model_index, "Best Weighted F1-Score"]

print(f"Best Model: {best_model_name}")
print(f"Best Weighted F1-Score: {best_model_f1:.4f}")
print(f"Best Parameters: {best_model_params}")


# In[204]:


# Combine training and validation sets for final training
X_train_final = pd.concat([X_train_scaled, X_val_scaled])
y_train_final = pd.concat([y_train, y_val])

# Train the CatBoost model with the best parameters
best_catboost_model = CatBoostClassifier(verbose=0, random_state=42, **best_model_params)
best_catboost_model.fit(X_train_final, y_train_final)

# Evaluate on test set
y_test_pred = best_catboost_model.predict(X_test_scaled)
weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Weighted F1-Score on Test Set: {weighted_f1:.4f}")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_test_pred))       


# In[207]:


# Generate the classification report as a dictionary
report = classification_report(y_test, y_test_pred, target_names=le.classes_, output_dict=True)

# Convert the report into a DataFrame for visualization
report_df = pd.DataFrame(report).T

# Remove support column
if 'support' in report_df.columns:
    report_df = report_df.drop(columns=['support'])

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :], annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.title("Classification Report Heatmap")
plt.xlabel("Metrics")
plt.ylabel("Classes")
plt.show()


# In[206]:


# Save the best model
best_model_path = r"C:\Users\User\ml-zoomcamp\capstone_1\models\best_model.pkl"
joblib.dump(best_catboost_model, best_model_path)
print(f"Best Model saved at: {best_model_path}")

# Save the processed datasets for reproducibility during inference
X_train_path = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed\X_train.pkl"
X_val_path = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed\X_val.pkl"
X_test_path = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed\X_test.pkl"
y_train_path = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed\y_train.pkl"
y_val_path = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed\y_val.pkl"
y_test_path = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed\y_test.pkl"

joblib.dump(X_train_scaled, X_train_path)
joblib.dump(X_val_scaled, X_val_path)
joblib.dump(X_test_scaled, X_test_path)
joblib.dump(y_train, y_train_path)
joblib.dump(y_val, y_val_path)
joblib.dump(y_test, y_test_path)

print("Processed datasets saved successfully in:", processed_data_folder)   

# Save the best parameters and metrics for reproducibility during inference
best_params_path = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed\best_params.pkl"
best_f1_path = r"C:\Users\User\ml-zoomcamp\capstone_1\data\processed\best_f1.pkl"
joblib.dump(best_model_params, best_params_path)
joblib.dump(best_model_f1, best_f1_path)

print(f"Best Parameters saved at: {best_params_path}")
print(f"Best F1-Score saved at: {best_f1_path}")

print("Best parameters and metrics saved successfully in:", processed_data_folder)

