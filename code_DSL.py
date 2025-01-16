import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline


def main():
    # Loading datasets
    data_dev = pd.read_csv('datasets/development_features.csv')
    data_eval = pd.read_csv('datasets/evaluation_features.csv')

    Xdata_dev = data_dev.drop(columns=['path', 'Id', 'age', 'num_words'])
    ydata_dev = data_dev['age']
    Xdata_eval = data_eval.drop(columns=['path', 'Id', 'num_words'])

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(Xdata_dev)
    Xdata_dev_scaled = scaler.transform(Xdata_dev)
    Xdata_eval_scaled = scaler.transform(Xdata_eval)
    
    # Dividing the dataset into development and validation
    X_train, X_validation, y_train, y_validation = train_test_split(Xdata_dev_scaled, ydata_dev, test_size=0.2, random_state=42)
    
    
    # Provo a rifare tutto con il Random Forest Regressor
    model = RandomForestRegressor(max_depth= 20, n_estimators= 200)

    best_numfeat = 95
    best_score = 50

    # # find the best features
    # for numfeat in range(85, len(Xdata_dev.columns), 5):
    #     rfe = RFE(model, n_features_to_select= numfeat)  
    #     rfe.fit(X_train, y_train)
        
    #     X_train_selected_forest = rfe.transform(X_train)
    #     X_test_selected_forest = rfe.transform(X_validation)

    #     # Fit the model
    #     model.fit(X_train_selected_forest, y_train)

    #     # Evaluate the model
    #     score = root_mean_squared_error(y_validation, model.predict(X_test_selected_forest))
    #     print(f'numfeat = {numfeat} , RMSE = {score} ')
        
    #     if score < best_score:
    #         best_score = score
    #         best_numfeat = numfeat
            
    # print(f'The best score is obtained for numfeat = {best_numfeat} and its value is {best_score}')  


    # Determining the MASK
    model = RandomForestRegressor(max_depth= 20, n_estimators= 200)
    rfe = RFE(model, n_features_to_select= best_numfeat, verbose=1, step=5)  

    mask_RF = rfe.support_
    features_used_RF = Xdata_dev.columns[mask_RF].to_list()
    print(features_used_RF)  # the mask of the selected features

    X_train_selected_forest = rfe.transform(X_train)
    X_test_selected_forest = rfe.transform(X_validation)
    
    # Fit the model
    model.fit(X_train_selected_forest, y_train)

    # Evaluate the model
    score = root_mean_squared_error(y_validation, model.predict(X_test_selected_forest))
    print(f'numfeat = {best_numfeat} , RMSE = {score} ')
    
    
    # TUNING PARAMS
    models_and_params = {
        'RandomForestRegressor': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            }
        },
    }
    
    results = {}
    
    model_name = models_and_params.keys()[0]
    model_details = models_and_params.values[0]
    
    print(f"\nEseguendo GridSearchCV per {model_name}...")
    
    # Creare la grid search
    grid_search = GridSearchCV(
        estimator=model_details['model'],
        param_grid=model_details['params'],
        cv=5,  # Numero di fold per la cross-validation
        scoring='neg_mean_squared_error',  # Usare MSE come metrica di valutazione
        n_jobs=-1
    )
    
    # Addestrare il modello
    grid_search.fit(X_train_selected_forest, y_train)
    
    # Salvare i risultati
    results[model_name] = {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_  # Invertire il segno per MSE
    }
    
    print(f"Migliori parametri per {model_name}: {grid_search.best_params_}")
    print(f"Miglior MSE (cross-val): {results[model_name]['best_score']:.4f}")

    # Valutazione sui dati di test
    y_pred = grid_search.best_estimator_.predict(X_validation[:, mask_RF])
    rmse_test = root_mean_squared_error(y_validation, y_pred)
    r2 = r2_score(y_validation, y_pred)

    print(f"RMSE sui dati di test: {rmse_test:.4f}")
    print(f"R^2 sui dati di test: {r2:.4f}")



    # PREDICTION
    model_pred = grid_search.best_estimator_
    Xdata_dev_masked = Xdata_dev_scaled[:, mask_RF]
    X_data_eval_masked = Xdata_eval_scaled[:, mask_RF]

    model_pred.fit(Xdata_dev_masked, ydata_dev)
    y_predicted = model_pred.predict(X_data_eval_masked)

    # Writing the csv file
    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Predicted'])
        for id_eval, age_pred in zip(data_eval['Id'], y_predicted):
            writer.writerow([id_eval, age_pred])
    
    

if __name__ == '__main__': 
    main()