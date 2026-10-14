# -*- coding: utf-8 -*-
"""
@author: ywgao
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from apyori import apriori
from kneed import KneeLocator
from scipy.spatial.distance import euclidean
from sklearn import metrics, model_selection, tree
from sklearn.cluster import KMeans
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

############################################################################
################## Supervised Data Mining Evaluation Func ##################
############################################################################

# =============================================================================
# 0.1.1. Classification Model Evaluations Using Confusion Matrix
# =============================================================================
def class_eval_num (y_true, y_pred):
    # make sure you import the metrics from sklearn before you run the function
    
    # y_true is the ground truth class
    # y_pred is the predicted class

    # Calculate confusion matrix
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Get the value for the four cells in the confusion matrix
    TN, FP, FN, TP = conf_matrix.ravel()
    
    # Calculate accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print("Accuracy:", "%.3f" % accuracy)

    # Calculate precision
    precision = metrics.precision_score(y_true, y_pred)
    print("Precision:", "%.3f" % precision)

    # Calculate sensitivity or recall
    sensitivity = metrics.recall_score(y_true, y_pred)
    print("Sensitivity or Recall:", "%.3f" % sensitivity)
      
    # Calculate specificity
    specificity = TN / (TN + FP)
    print("Specificity:", "%.3f" % specificity)


# =============================================================================
# 0.1.2. Classification Model Evaluations Using ROC Curve
# =============================================================================
def class_eval_fig (y_true, y_prob):
    # make sure you import the metrics from sklearn and pyplot before you 
    # run the function

    # y_true is the ground truth class 
    # y_pred is the predicted probability of class 1

    # Compute ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    
    # Compute ROC AUC score
    roc_auc = metrics.roc_auc_score(y_true, y_prob)
    print("ROC AUC:", roc_auc) 
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


# =============================================================================
# 0.2. Prediction Model Evaluations (Dependent variable is a numerical variable)
# =============================================================================
def prd_evl (y_true, y_pred, k):
    # y_true is the ground truth value of dependent variable
    # y_pred is the predicted value of dependent variable
    # k is number of predictors (independent variables)

    # Mean Error (ME)
    me = np.mean(y_true - y_pred)

    # Mean Percentage Error (MPE)
    mpe = np.mean((y_true - y_pred)/y_true *100)

    # Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R-squared (R2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Adjusted R-squared (adjusted R2)
    n = len(y_true)

    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    
    # Display the results
    print("Mean Error (ME):", "%.3f" % me)
    print("Mean Percentage Error (MPE):", "%.3f" % mpe)
    print("Mean Absolute Deviation (MAD):", "%.3f" % mad)
    print("Mean Squared Error (MSE):", "%.3f" % mse)    
    print("Root Mean Squared Error (RMSE):", "%.3f" % rmse)
    print("Mean Absolute Percentage Error (MAPE):", "%.3f" % mape)
    print("R-squared (R2):", "%.3f" % r2)
    print("Adjusted R-squared (adjusted R2):", "%.3f" % adjusted_r2)

############################################################################
################## Supervised Data Mining Model Functions ##################
############################################################################

# =============================================================================
# # 1.1.Linear Regression Model (Numerical Dependent Variable)
# =============================================================================
def linear_reg(X, y, new_X=None):
    ## Count number of input variables
    X_col = list(X.columns)
    y_col = list(y.columns)
    X_num = len(X.columns)

    ## Step 1. Preprocess X and y
    X = X.astype(int)
    y = y[y_col[0]]
    X, y = np.array(X), np.array(y)
    X = sm.add_constant(X)
    
    ## Step 2. Create and train linear regression model
    model = sm.OLS(y, X) 
    model_output = model.fit()
    
    if new_X is None:
        output1 = "******Output for Step 3. Run Linear Regression Model and check its performance******"
        print('\n')
        print(len(output1)*"*")
        print(output1)
        print(len(output1)*"*")
        print("The model includes independent variables:", X_col)
        print("The model includes dependent variable:", y_col)
        ## Step 3. Get results 
        # regression model output
        print("\n******Below is the regression output******\n")
        print(model_output.summary())
        
        ## Step 4. Evaluate model performance
        # check the R-square and adjusted R-square in the above output
        # alternative: use the function we defined previously
        print("\n******Below is the evaluation output******\n")
        y_pred = model_output.predict(X)
        prd_evl(y, y_pred, X_num)
    
    ## Step 5. Predict value if new data is provided
    if new_X is not None:
        output2 = "******Output for Step 4. Prediction using Linear Regression Model******"
        print('\n')
        print(len(output2)*"*")
        print(output2)
        print(len(output2)*"*")
        print("The prediction model includes independent variables:", X_col)
        print("The prediction model includes dependent variable:", y_col)
        new_X = new_X.astype(int)
        new_X = np.array(new_X)
        new_X = sm.add_constant(new_X)
        print("The predicted value is returned")
        return(model_output.predict(new_X))
    

# =============================================================================
# # 1.2.Linear Probability Model (Categorical Dependent Variable)
# =============================================================================
def lpm(X, y, new_X=None):
    X_col = list(X.columns)
    y_col = list(y.columns)
    ## Step 1. Preprocess X and y
    X = X.astype(int)
    X, y = np.array(X), np.array(y)
    X = sm.add_constant(X)
    
    ## Step 2. Create and train "linear probability model"
    model = sm.OLS(y, X) 
    model_output = model.fit()

    if new_X is None:
        output1 = "******Output for Step 3. Run Linear Probability Model and check its performance******"
        print('\n')
        print(len(output1)*"*")
        print(output1)
        print(len(output1)*"*")
        print("The model includes independent variables:", X_col)
        print("The model includes dependent variable:", y_col)
        ## Step 3. Get results 
        # Linear regression model output
        print("\n******Below is the regression output******\n")
        print(model_output.summary())
        ## Step 4. Step Evaluate model performance
        # check the R-square and adjusted R-square in the above output
        # alternative: use the function we defined previously
        print("\n******Below is the evaluation output******\n")
        y_pred = model_output.predict(X) > 0.5
        class_eval_num(y, y_pred)
        class_eval_fig(y, y_pred)
    
    ## Step 5. Predict value if new data is provided
    if new_X is not None:
        new_X = new_X.astype(int)
        new_X = np.array(new_X)
        new_X = sm.add_constant(new_X)
        output2 = "******Output for Step 4. Prediction using Linear Probability Model******"
        print('\n')
        print(len(output2)*"*")
        print(output2)
        print(len(output2)*"*")
        print("The prediction model includes independent variables:", X_col)
        print("The prediction model includes dependent variable:", y_col)
        print("\nThe predicted category is returned")
        return((model_output.predict(new_X)>=0.5)*1)


# =============================================================================
# # 1.3.Logistic Regression Model (Categorical Dependent Variable)
# =============================================================================
def logit(X, y, new_X=None):
    X_col = list(X.columns)
    y_col = list(y.columns)
    ## Step 1. Preprocess X and y
    X = X.astype(int)
    X, y = np.array(X), np.array(y)
    X = sm.add_constant(X)
    
    ## Step 2. Create and train "logistic regression model"
    model = sm.Logit(y, X) 
    model_output = model.fit()

    if new_X is None:
        output1 = "******Output for Step 3. Run Logistic Regression Model and check its performance******"
        print('\n')
        print(len(output1)*"*")
        print(output1)
        print(len(output1)*"*")
        print("The model includes independent variables:", X_col)
        print("The model includes dependent variable:", y_col)
        ## Step 3. Get results of Logit Regression Model
        print("\n******Below is the regression output******\n")
        print(model_output.summary())
        
        ## Step 4. Evaluate model performance
        # check the Pseudo R-square
        # alternative: use the function we defined previously
        print("\n******Below is the evaluation output******\n")
        y_pred = model_output.predict(X) > 0.5
        class_eval_num(y, y_pred)
        class_eval_fig(y, y_pred)
    
    ## Step 5. Predict value if new data is provided
    if new_X is not None:
        new_X = new_X.astype(int)
        new_X = np.array(new_X)
        new_X = sm.add_constant(new_X)
        output2 = "******Output for Step 4. Prediction using Logistic Regression Model******"
        print('\n')
        print(len(output2)*"*")
        print(output2)
        print(len(output2)*"*")
        print("The prediction model includes independent variables:", X_col)
        print("The prediction model includes dependent variable:", y_col)
        print("\nThe predicted category is returned")
        return((model_output.predict(new_X)>0.5)*1)


# =============================================================================
# # 2. Naive Bayes Model (Categorical Dependent Variable)
# =============================================================================
def naive_bayes(X, y, new_X=None):
    X_col = list(X.columns)
    y_col = list(y.columns)
    
    ## Step 1. Preprocess X and y
    X, y = np.array(X), np.array(y)
    X = OrdinalEncoder().fit_transform(X)
    
    ## Step 2. Create and train categorical naive bayes model
    model = CategoricalNB()
    model.fit(X, y)

    if new_X is None:
        output1 = "******Output for Step 3. Run Naive Bayes Model and check its performance******"
        print('\n')
        print(len(output1)*"*")
        print(output1)
        print(len(output1)*"*")
        print("The model includes independent variables:", X_col)
        print("The model includes dependent variable:", y_col)
        ## Step 3 Evaluate model performance
        # accuracy evaluation based on the cross-validation
        print("\n******Below is the evaluation output******\n")
        print("Average accuracy rate is", model_selection.cross_val_score(model, X, y, cv=5, scoring = "accuracy").mean())
        # evaluation based on the function we create
        y_train_pred = model_selection.cross_val_predict(model, X, y, cv=5)
        class_eval_num(y, y_train_pred)
        class_eval_fig(y, y_train_pred)
    
    if new_X is not None:
        output2 = "******Output for Step 4. Prediction using Naive Bayes Model******"
        print('\n')
        print(len(output2)*"*")
        print(output2)
        print(len(output2)*"*")
        ## Step 4. Predict value if new data is provided
        new_X = np.array(new_X)
        new_X = OrdinalEncoder().fit_transform(new_X)
        y_pred = model.predict(new_X)
        print("The prediction model includes independent variables:", X_col)
        print("The prediction model includes dependent variable:", y_col)
        print("\nThe predicted category is returned")
        return(y_pred)


# =============================================================================
# # 3. K-Nearest Neighbors (Categorical Dependent Variable)
# =============================================================================
def KNN(X, y, start_k, end_k, optimal_k=None, new_X=None):
    X_col = list(X.columns)
    y_col = list(y.columns)
    ## Step 1. Preprocess X and y
    X, y = np.array(X), np.array(y)
    X = StandardScaler().fit_transform(X)
       
    if new_X is None:
        output1 = "******Output for Step 3. Run K-Nearest Neighbors Model and check its performance******"
        print('\n')
        print(len(output1)*"*")
        print(output1)
        print(len(output1)*"*")
        print("The model includes independent variables:", X_col)
        print("The model includes dependent variable:", y_col)
        ## Step 3 & 4. Create, train and evaluate KNN model with different number of k
        print("\n******Below is the evaluation output. Please select the optimal_k value based on the output below******\n")
        for k in range(start_k, end_k):
            ## Step 3. Create and train KNN model with different number of k    
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X, y)
            ## Step 4. Evaluate model performance
            # accuracy evaluation based on the cross-validation
            k_acc = model_selection.cross_val_score(model, X, y, cv=5, scoring = "accuracy")
            print("k = " + str(k), "accuracy rate is", "%.2f" % k_acc.mean())
# =============================================================================
#             # evaluation based on the function we create, unnote this if needed
#             y_train_pred = model_selection.cross_val_predict(model, X, y, cv=5)
#             class_eval_num(y, y_train_pred)
#             class_eval_fig(y, y_train_pred)
#             print("\n\n")
# =============================================================================

    if new_X is not None: ## Use optimal k
        output2 = "******Output for Step 4. Prediction using K-Nearest Neighbors Model******"
        print('\n')
        print(len(output2)*"*")
        print(output2)
        print(len(output2)*"*")
        print("The prediction model includes independent variables:", X_col)
        print("The prediction model includes dependent variable:", y_col)
        ## Step 5. Predict value if new data is provided
        new_X = np.array(new_X)
        new_X = StandardScaler().fit_transform(new_X)
        print("\nThe predicted category is returned using the optimal k = ", optimal_k)
        model = KNeighborsClassifier(n_neighbors=optimal_k)
        model.fit(X, y)
        y_pred = model.predict(new_X)
        return(y_pred)

# =============================================================================
# # 4.1.Decision Tree - Classification Tree (Categorical Dependent Variable)
# =============================================================================
def classification_tree(X, y, max_depth, min_samples_split, new_X=None):
    X_col = list(X.columns)
    y_col = list(y.columns)
    ## Step 1. Preprocess X and y
    X_dummy = pd.get_dummies(X)
    featureNames = X_dummy.columns
    X, y = np.array(X_dummy), np.array(y)
       
    ## Step 2. Create and train Classification Tree model
    model = DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state=42)
    model.fit(X, y)

    if new_X is None: 
        output1 = "******Output for Step 3. Run Classification Tree Model and check its performance******"
        print('\n')
        print(len(output1)*"*")
        print(output1)
        print(len(output1)*"*")
        print("The model includes independent variables:", X_col)
        print("The model includes dependent variable:", y_col)
        ## Step 3. Get results of Classification Tree
        fig, ax = plt.subplots(figsize = (40,20))
        tree.plot_tree(model, node_ids = True, proportion = True, impurity = False, fontsize=20, feature_names = featureNames, class_names = ['0','1'], rounded = True, filled = True)
        plt.show()
        print("\n*****The classification tree output is returned in the `Plot`*****\n")
        ## Step 4. Evaluate model performance
        print("******Below is the evaluation output******\n")
        # accuracy evaluation based on the cross-validation
        model_selection.cross_val_score(model, X, y, cv=5, scoring = "accuracy")
        # evaluation based on the function we create
        y_train_pred = model_selection.cross_val_predict(model, X, y, cv=5)
        class_eval_num(y, y_train_pred)
        class_eval_fig(y, y_train_pred)

    ## Step 5. Predict value if new data is provided
    # DO NOT need to use k-fold cross-validation because we're not training and 
    # evaluating  model now, but we try to make prediction now
    if new_X is not None: 
        new_X = pd.concat([X_dummy, pd.get_dummies(new_X)])[len(X_dummy):].fillna(False)
        # new_X = pd.get_dummies(new_X)
        new_X = np.array(new_X)
        output2 = "******Output for Step 4. Prediction using Classification Tree Model******"
        print('\n')
        print(len(output2)*"*")
        print(output2)
        print(len(output2)*"*")
        print("The prediction model includes independent variables:", X_col)
        print("The prediction model includes dependent variable:", y_col)
        print("\nThe predicted category is returned.")
        y_pred = model.predict(new_X)
        return(y_pred)
        

# =============================================================================
# # 4.2.Decision Tree - Regression Tree (Numerical Dependent Variable)
# =============================================================================
def regression_tree(X, y, max_depth, min_samples_split, new_X=None):
    X_col = list(X.columns)
    y_col = list(y.columns)
    ## Step 1. Preprocess X and y
    X_dummy = pd.get_dummies(X)
    featureNames = X_dummy.columns
    y = y[y_col[0]]
    X, y = np.array(X_dummy), np.array(y)
    
    ## Step 2. Create and train Classification Tree model
    model = DecisionTreeRegressor(max_depth = max_depth, min_samples_split = min_samples_split, random_state=42)
    model.fit(X, y)

    if new_X is None: 
        output1 = "******Output for Step 3. Run Regression Tree Model and check its performance******"
        print('\n')
        print(len(output1)*"*")
        print(output1)
        print(len(output1)*"*")
        print("The model includes independent variables:", X_col)
        print("The model includes dependent variable:", y_col)
        ## Step 3. Get results of Classification Tree
        fig, ax = plt.subplots(figsize = (40,20))
        tree.plot_tree(model, node_ids = True, proportion = True, impurity = False, fontsize=20, feature_names = featureNames, rounded = True, filled = True)
        plt.show()
        print("\n*****The regression tree output is returned in the `Plot`*****\n")
        ## Step 4. Evaluate model performance
        # evaluation based on the function we create
        print("******Below is the evaluation output******\n")
        y_train_pred = model_selection.cross_val_predict(model, X, y, cv=5)
        prd_evl(y, y_train_pred, len(featureNames))

    if new_X is not None: 
    ## Step 5. Predict value if new data is provided
        new_X = pd.concat([X_dummy, pd.get_dummies(new_X)])[len(X_dummy):].fillna(False)
        # new_X = pd.get_dummies(new_X)
        new_X = np.array(new_X)
        output2 = "******Output for Step 4. Prediction using Regression Tree Model******"
        print('\n')
        print(len(output2)*"*")
        print(output2)
        print(len(output2)*"*")
        print("The prediction model includes independent variables:", X_col)
        print("The prediction model includes dependent variable:", y_col)
        print("\nThe predicted value is returned.")
        y_pred = model.predict(new_X)
        return(y_pred)

############################################################################
################# Unsupervised Data Mining Model Functions #################
############################################################################

# =============================================================================
# 1. Cluster Analysis
# =============================================================================
# =============================================================================
# The following is an example to run a cluster analysis to categorize the 
# airbnb into different gorups based on their features
#     * Step 1: Preprocess data (columns)
#     * Step 2: Specify the X 
#     * Step 3. Train cluster analysis using multiple k to select optimal k
#     * Step 4. Train cluster analysis using the optimal k
#     * Step 5. Get and interpret the results 
# =============================================================================
def cluster_kmeans(X, start_k, end_k, optimal_k=None):
    # Step 1. Define the input of X
    # StandardScaler standardize the data, which is required by cluster analysis 
    # because cluster analysis is based on similarity calculation
    X_list = list(X.columns)
    X = StandardScaler().fit_transform(X)
    
    # Step 2. Select the optimal number of clusters based on SSE curve in SSE Elbow Plot
    # Use both domain knowledge and clustering metrics to decide how many clusters to create.
    if optimal_k is None:
        sse = []
        for k in range(start_k, end_k):
            model = KMeans(n_clusters=k, init = "random", n_init = 1, max_iter = 200, random_state = 40)
            model.fit(X)
            sse.append(model.inertia_)
        plt.plot(range(start_k, end_k), sse, marker='o')
        plt.title('Elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.show()    
        # If you have difficulty to decide the elbow point, use the function below
        kl = KneeLocator(range(start_k, end_k), sse, curve="convex", direction="decreasing")
        print("Based on SSE Elbow Plot, the optimal k should be:", kl.elbow)
    
    # Step 3&4. Create and train cluster analysis model using optimal k and get results
    if optimal_k is not None:
        # Step 3. Create and train cluster analysis model using optimal k
        model = KMeans(n_clusters = optimal_k, n_init = 1, max_iter = 200, random_state = 40)
        model.fit(X)
        
        # Step 4. Get results 
        print("\n", "Sizes of clusters")
        cluster_size = pd.Series(model.labels_).value_counts().sort_index()
        cluster_size.index = ['Cluster '+ str(each+1) for each in cluster_size.index]
        print(cluster_size)
        
        # Find locations of the centroids (means)
        print("\n\n", "Average of variables in each cluster")
        centroids = pd.DataFrame(model.cluster_centers_, columns = X_list)
        centroids.index = ['Cluster '+ str(each+1) for each in centroids.index]
        print(centroids)
        
        # Find cohension of each cluster
        cohesion_scores = {}
        for cluster_id in np.unique(model.labels_):
            # Airbnb data
            cluster_raw_points = X[model.labels_ == cluster_id]
            # Centroid data
            cluster_centroid = model.cluster_centers_[cluster_id]
            # Claculate distance between each row of the data and centroid data
            distances = [euclidean(point, cluster_centroid) for point in cluster_raw_points]
            # Calculate the average distance (cohesion) for the cluster
            cohesion_scores[f'Cluster {cluster_id+1}'] = np.mean(distances)
        cohesion_scores = pd.DataFrame([cohesion_scores]).transpose()
        print("\n\n", "Within-cluster sum of squares (WSS):", cohesion_scores)
        return(model.predict(X))

# =============================================================================
# 2. Association Rule
# =============================================================================
def association_rule(data, min_support, min_confidence, min_lift):
    ## Step 1. Run association rule analysis model 
    # Run the apriori algorithm - this is what computes the association rules.
    # CHANGE THE PARAMETERS BELOW!!
    # min_support -    the minimum threshold for support; below this the rule won't appear in the output
    # min_confidence -   the minimum threshold for confidence; below this the rule won't appear in the output
    # min_lift -    the minimum threshold for lift; below this the rule won't appear in the output
    aRules = apriori(data, min_support = min_support, min_confidence = min_confidence, min_lift = min_lift)
    
    ## Step 2. Reorganize results to a readable one
    # This outputs the list of rules to the screen and writes them to a file.
    # It's easier to see them if you just open the output file in Excel. You can also use Excel
    #    to sort the rules by confidence, support, and lift
    results_df = pd.DataFrame(list(aRules))
    results_df = results_df.explode(['ordered_statistics']).reset_index()
    results_df['antecedent'] = results_df['ordered_statistics'].apply(lambda x: x[0])
    results_df['consequent'] = results_df['ordered_statistics'].apply(lambda x: x[1])
    results_df['confidence'] = results_df['ordered_statistics'].apply(lambda x: x[2])
    results_df['lift'] = results_df['ordered_statistics'].apply(lambda x: x[3])
    results_df[['antecedent', 'consequent']] = results_df[['antecedent', 'consequent']].map(lambda x: "{" + ", ".join(x) + "}")
    results_df['rules'] = results_df[['antecedent', 'consequent']].agg("->".join, axis=1)
    results_df = results_df[['rules', 'support', 'confidence', 'lift']].sort_values('lift', ascending = False).reset_index(drop=True)
    
    ## Step 3. Get results
    print(results_df)
    return(results_df)
    # you can also save it as a csv or excel file
    # results_df.to_csv("T7.AssociationRule_Output.csv")
