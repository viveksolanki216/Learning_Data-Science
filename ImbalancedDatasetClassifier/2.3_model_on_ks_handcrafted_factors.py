
# This model is an extenstion of the model "Learnings_Sourced_v_Not_HighScoring" that I have worked on in the past.
    # What has been done in the project "Learnings_Sourced_v_Not_HighScoring" in Sep2024-Feb2025:
    # The code is directly taken from the v3 code of "" dir,
    # python file, score_allmult_3_AllDeals.py.
    # Objective: To get the model and predict the sourced vs non-sourced deals.
    # Data: All multiple, full kitchen sink's rounds data.
        # Experimentation: 1) Training on high scoring 2) All Deals.
    # Factors: all factors that starts with f_
        # Experimentation: 1) Aff factors, 2) Aff full kitchen sink's factors that falls in the set/groupings provided
                         # 3) Adding "score_overall" to above.
    # Model: 1) Random Forest, 2) Logistic Regression. Will be using Random Forest.
    # Model Evaluation:
    # Analysis v/s scoring:
        # Analysis: In analysis, we are mostly assessing for feature importance. We first selec top k features from
                   # each set/grouping of features using random forest training for that set of factors.
                   # And then we use these set of top k*n features (n=number of sets) to check the order in which the factors
                    # are getting entered in the model using SequentialFeatureSelector with LinearRegressin model.
        # Scoring: We just throw all the factors in the model, assess the CV performance. And score the allmult rounds.


    # Observations: We saw that various factors are able to distinguish between sourced and non-sourced deals.
                  # that says that our sourced deals is not just a random selection of industry deals.
                  # There are bias in selections.

# Why the above model is interesting for sourcing team.
    # Moiz has told that the selection model performs well on the sourced deals v/s industry deals. and better for the portfolio deals.
    # Since sourced deals 5000 set is smaller and assessing performance for the selection model could be variable/in-robust.
    # So that want a broader set of the sourced deals. And the above model could come in pciture where we can guess
    # some similar looking deals as sourced from the model and then assess the performance of the selection model on these deals.

# The model is renamed to Accessibility_Model.

# What's new here?
    # Adding more hand-crafted factors and assess model performance.

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
sys.path.append(os.getcwd() + "/Accessibility_Model/Modelling/src")
from utils import *
sys.path.append('/Users/vss/Work/R_And_Python_CV_Utilities')
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid
from scipy.stats import uniform
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import Pipeline

if __name__ == '__main__':

    # Dir / File Paths
    main_dir = os.getcwd() + "/Accessibility_Model/Modelling/"
    in_dir = f"{main_dir}input/"
    out_dir = f"{main_dir}output/"
    tmp_dir = f"{main_dir}tmp/"

    aff_train = pd.read_pickle(f'{tmp_dir}/aff_train_new_features.pkl')

    #aff_to_score = pd.read_pickle(f'{tmp_dir}/aff_to_score.pkl')
    print(aff_train['sourced?'].value_counts())
    print(aff_train.groupby('cv_k')['sourced?'].value_counts())

    target_labels = ['sourced?']
    FEATURES =  [feature for feature in aff_train.columns if feature.startswith('f_')]
    print(len(FEATURES))
    print(aff_train[FEATURES].isna().sum().sort_values())

    roundids = aff_train['roundid']
    cv_k = aff_train['cv_k']
    X = aff_train[FEATURES].values
    y = aff_train[target_labels].reset_index(drop=True).values.ravel()
    print(y.sum(), y.mean())

    # select only top 250 features
    params = {'n_estimators': 100, 'min_samples_leaf': 100, 'max_features': 50, 'max_depth':7, 'class_weight': {0: 0.1, 1: 0.9}, 'random_state':0, 'n_jobs':-1}
    model_rf = RandomForestClassifier(**params)
    model_rf, probs_train_rf, probs_test_rf = train_and_score(X, y, X, y, model_rf, oversample=False)
    feature_importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': model_rf.feature_importances_
       })
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    FEATURES = feature_importance['feature'].tolist()[0:250]

    # Update the X
    X = aff_train[FEATURES].values


    # Exmaple to over sample the data
    #oversample = SMOTE(sampling_strategy=1, random_state=0, k_neighbors=5)
    #X_ovr, y_ovr = oversample.fit_resample(pd.DataFrame(X).fillna(-1).values, y)
    #print(y_ovr.mean())

    #model_rf = RandomForestClassifier()
    #distributions = dict(
    ##    n_estimators=[100],
    #    max_depth=[5, 10, 20, 50],
        #min_samples_leaf=[100, 200, 500],
    #    class_weight=[{0: 0.1, 1: 0.9}], #{0: 0.25, 1: 0.75}, {0: 0.167, 1: 0.833},
    #    max_features=[50],
    #)
    #clf = RandomizedSearchCV(
    #    model_rf, distributions, random_state=0, cv=4, scoring='average_precision', # equivalent to average_precision
    #    n_jobs=-1, n_iter=15, verbose=3, return_train_score=True
    #)
    #search = clf.fit(
    #    pd.DataFrame(X).fillna(-1), y,
    #)
    #grid_search_summary = pd.DataFrame(search.cv_results_).sort_values('rank_test_score').reset_index(drop=True)

    #print("Best Params: ", grid_search_summary['params'][0])
    #print("worst Params: ", grid_search_summary['params'][14])

    distributions = dict(
        n_estimators=[200],
        max_depth= [40],#[3, 5, 7, 10, 20, 30, 40, 50],
        min_samples_leaf= [2],
        sampling_strategy=['all'],
        replacement=[False],
        bootstrap=[True],
        max_samples=[5000],
        class_weight=["balanced_subsample"],#[{0: 0.5, 1: 0.5}],#, {0: 0.167, 1: 0.833},
        max_features=['sqrt']#[50],
    )
    param_grid = ParameterGrid(distributions)
    #params = {'n_estimators': 150, 'min_samples_leaf': 100, 'max_features': 100, 'class_weight': {0: 0.1, 1: 0.9}}
    #params = {'n_estimators': 100, 'min_samples_leaf': 100, 'max_features': 50, 'class_weight': {0: 0.1, 1: 0.9}}
    #best_params = {'n_estimators': 100, 'min_samples_leaf': 30, 'max_features': 100, 'class_weight': {0: 0.1, 1: 0.9}}
    #worst_params = {'n_estimators': 10, 'min_samples_leaf': 10, 'max_features': 25, 'class_weight': {0: 0.167, 1: 0.833}}
    #params = {'n_estimators': 100, 'min_samples_leaf': 50, 'max_features': 50}

    # Just use best parameters from the 2.1_model_on_ks_factors.py
    # Cross Validate perforamnce of the model
    gridsearch_Results = []
    decile_pref_list = []
    for params in param_grid:
        #params = param_grid[0]
        params_add = {'n_jobs':-1, 'random_state': 0}
        params.update(params_add)
        print(params)

        #model_rf = RandomForestClassifier(**params)
        model_rf = BalancedRandomForestClassifier(**params)
        # Cross Validation Testing.
        decile_perf, cv_perf_summary, final_prediced_data = run_model_on_4_cv(model_rf, X, y, cv_k, roundids, oversample=False, train_size=None)

        train_test_results = cv_perf_summary.groupby('train or test').mean().unstack()
        train_test_results.index = [ f'{b} {a}' for a, b in train_test_results.index]
        train_test_results['params'] = str(params)
        gridsearch_Results.append(train_test_results)

        decile_pref_list.append(decile_perf)


    gridsearch_Results1 = pd.DataFrame(gridsearch_Results)

    with pd.ExcelWriter(out_dir+"run3/hp-tune-imbal-Top250F_undersample2.xlsx", engine="openpyxl") as writer:
        gridsearch_Results1.to_excel(writer, 'exp', index=False)


    # Learning Curve
    params = {'n_estimators': 100, 'min_samples_leaf': 20, 'max_depth': 10, 'max_features': 'sqrt', 'class_weight': 'balanced', 'n_jobs':-1, 'random_state': 0}
    train_sizes=np.arange(10000, 50000, 10000).tolist()
    gridsearch_Results = []
    decile_pref_list = []
    for train_size in train_sizes:
        print(train_size)
        #params = param_grid[0]
        model_rf = RandomForestClassifier(**params)
        # Cross Validation Testing.
        decile_perf, cv_perf_summary, final_prediced_data = run_model_on_4_cv(model_rf, X, y, cv_k, roundids, oversample=False, train_size=train_size)

        train_test_results = cv_perf_summary.groupby('train or test').mean().unstack()
        train_test_results.index = [ f'{b} {a}' for a, b in train_test_results.index]
        train_test_results['params'] = str(params)
        train_test_results['train_size'] = train_size
        gridsearch_Results.append(train_test_results)

        decile_pref_list.append(decile_perf)


    gridsearch_Results1 = pd.DataFrame(gridsearch_Results)

    with pd.ExcelWriter(out_dir+"run3/hp-tune-imbal-Top250F_SMOTE33_2.xlsx", engine="openpyxl") as writer:
        gridsearch_Results1.to_excel(writer, 'Max-depth', index=False)

    # Train on full set for feature importance and score other rounds
    model_rf, probs_train_rf, probs_test_rf = train_and_score(X, y, X, y, model_rf)

    feature_importance = pd.DataFrame({
        'if new feature?': [1 if feature in handcrafted_factors else 0 for feature in FEATURES],
        'feature': FEATURES,
        'importance': model_rf.feature_importances_
       })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    feature_importance['importance_cumsum'] = feature_importance['importance'].cumsum()
    feature_importance = feature_importance.reset_index(drop=True)
    new_feature_importance = feature_importance[feature_importance['if new feature?'] == 1]
    feature_importance.loc[feature_importance.index < 100,'if new feature?'].value_counts()

    with pd.ExcelWriter(out_dir+"run3/v2-sourced_v_not_using_ksfactors_and_newfeatures.xlsx", engine="openpyxl") as writer:
        decile_perf.to_excel(writer, 'Combined_Perf_ValidSet_CV=4', index=False)
        cv_perf_summary.to_excel(writer, "CV=4 summary", index=False)
        final_prediced_data.to_excel(writer, "Pred val set", index=False)
        feature_importance.to_excel(writer, "Feature Importance", index=False)

    # Consolidated performance
    pd.DataFrame(FEATURES, model_rf.feature_importances_)

    # Single Train-Test Paradigm Testing
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # train_rf_perf, test_rf_perf = test_model_on_train_test_split(X_train, y_train, X_test, y_test, model_rf)
    # print(train_rf_perf)
    # print(test_rf_perf)
    # test rf model

    # Fit model on the full set.
    model_rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, max_depth=5,
                                      class_weight={0: 0.167, 1: 0.833},
                                      max_features=25, n_jobs=-1
    )
    model_rf.fit(X.fillna(X.mean()), y)
    probs_rf = model_rf.predict_proba(X_to_score.fillna(X.mean()))
    #get_decile_performance(probs_rf[:, 1], aff_ks_subset['sourced?'].values)
    feature_importances = pd.DataFrame({'Features': model_rf.feature_names_in_, 'score':model_rf.feature_importances_}).sort_values('score', ascending=False)

    # Fit model on the full set.
    model_lr = LogisticRegression(penalty='l2', C=0.25, solver='liblinear', max_iter=200,
                                  class_weight={0: 0.167, 1: 0.833})
    model_lr.fit(X.fillna(X.mean()), y)
    probs_lr = model_lr.predict_proba(X_to_score.fillna(X.mean()))
    #get_decile_performance(probs_lr[:, 1], aff_ks_subset['sourced?'].values)
    coefs = pd.DataFrame({
        'features': X_to_score.columns,
        'coefs l1 penalty':  model_lr.coef_.tolist()[0],
    }).set_index('features')


    probs = pd.DataFrame(np.c_[probs_rf[:, 1], probs_lr[:, 1]], columns=['prob sourced rf', 'prob sourced lr'])
    probs2 = pd.concat([aff_ks_subset[['roundid', 'sourced?']], probs], axis=1)
    probs2.insert(2, 'train instance?', np.where(aff_ks_subset['roundid'].isin(aff_train['roundid']), 1, 0))
    probs2 = probs2.drop_duplicates('roundid')

    allmult = pd.read_csv(
        "/Users/vss/Correlation Ventures Dropbox/Correlation Ventures/Analytics/vs_transition/Shoeb's Work/Data/16Sep2024/final_output/allMultiples2024Q2.csv")
    allmult = pd.merge(allmult[['roundid']], probs2, on='roundid', how='left')

    allmult2 = allmult[allmult['sourced?'].notna()]
    print(get_decile_performance(allmult2['prob sourced lr'], allmult2['sourced?'].values))
    print(get_decile_performance(allmult2['prob sourced rf'], allmult2['sourced?'].values))

    with pd.ExcelWriter(out_dir + 'v4.1(train_on_highscoring)-allmult_rounds_est_prob_sourced.xlsx') as writer:
        allmult.to_excel(writer, "probs", index=False)


