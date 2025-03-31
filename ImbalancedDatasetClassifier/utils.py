
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/vss/Work/R_And_Python_CV_Utilities')
from Python.read_data import read_aff
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc
from imblearn.over_sampling import SMOTE

def read_factors_groupin():
    file_path = '/Users/vss/Work/Learnings_Sourced_v_Not_HighScoring/input/sas/varlist.inc'

    dict = {}
    #values = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("%let"):
                # Capture the variable name
                var_name = line.split()[1]
                dict[var_name] = [] # Start a new list of values for this variable
            elif line and not line.startswith(';'):
                # Append values to the last list in 'values'
                dict[var_name].extend(line.replace(';', '').split())

    data = [(key, value) for key, values in dict.items() for value in values]
    # Convert to a DataFrame
    features_set = pd.DataFrame(data, columns=["Set", "feature"])

    return features_set

def save_aff_kitchen_sink_subset(kitchen_sink_file, out_dir):
    '''
    AFF kitchen sink is too large that takes time to load so save a version
    that has required features only.

    :param kitchen_sink_file:
    :param out_dir:
    :return:
    '''
    # Read features and grouping from the varlist.inc
    features_set = read_factors_groupin()
    # Use only those features from grouping suggested by Anu.
    features_groupings = [
        'v_geo', 'v_dist', 'v_vcexit', 'v_vcmult', 'v_active', 'v_synd', 'v_board', 'v_status', 'v_raised',
        'v_board_pd', 'v_vctm'
    ]
    features_set = features_set[features_set['Set'].isin(features_groupings)]

    # Read kitchen sink file, its huge file and takes time to load
    aff_ks = pd.read_csv(kitchen_sink_file, encoding='latin')  # , low_memory=False)
    FEATURES_ALL = [col for col in aff_ks.columns if col.startswith('f_')]

    features_set = features_set[features_set['feature'].isin(FEATURES_ALL)]
    features_set = features_set.drop_duplicates('feature')
    FEATURES = features_set['feature'].tolist()

    aff_ks_subset = aff_ks[['RoundID', 'year', 'Filters', 'score_overall'] + FEATURES]
    aff_ks_subset.to_csv(f'{out_dir}/aff_kitchen_sink_subset.csv', index=False)
    return "generated"

def prepare_data_v2(source_round_mapping_file, aff_ks_subset):
    '''
    Tags sourced deals in the AFF data using the source_round_mapping_file.
    Filter deals using Filters=1 and year>=2010 for apples-to-apples for non-sourced deals v sourced deals
    :param source_round_mapping_file:
    :param aff_ks_subset:
    :return:
    '''

    # Loading Source ID to Round ID mapping by Megha, total 3525 mappings found.
    # *Update File Needed?
    source_round_mapping = pd.read_excel(source_round_mapping_file)
    source_round_mapping = source_round_mapping[['Source#', 'Round ID']].dropna().rename(
        columns={'Source#': 'source_number', 'Round ID': 'roundid'})
    source_round_mapping.drop_duplicates(['roundid'], inplace=True)

    aff = aff_ks_subset
    # aff = pd.read_csv(aff_file, encoding='latin', low_memory=False)
    aff.rename(columns={'RoundID': 'roundid'}, inplace=True)

    aff = aff.merge(source_round_mapping, on='roundid', how='left')
    aff['sourced?'] = aff['source_number'].notnull().astype(int)
    print(aff['sourced?'].value_counts())

    FEATURES = [col for col in aff.columns if col.startswith('f_')]

    # High Scoring data, high scoring 320 sourced vs ~10000 non-sourced deals
    aff_sub = aff.loc[
        (aff['Filters'] == 1) & (aff['year'] >= 2010),
        ['roundid', 'score_overall', 'sourced?', 'source_number', 'year'] + FEATURES
    ].reset_index(drop=True)

    print(aff_sub['sourced?'].value_counts())
    print(np.round(aff_sub.groupby(['sourced?'])['year'].value_counts(normalize=True).unstack().T * 100,3))

    #aff_sub.to_csv(f"{in_dir}aff_sub.csv", index=False)

    return aff_sub, aff, source_round_mapping


def drop_features_with_high_missing(aff_sub: pd.DataFrame, perc_mising_theshold: int)-> pd.DataFrame:
    # Missing analysis, drop columns with high missing numbers
    n = perc_mising_theshold * aff_sub.shape[0]
    missings = aff_sub[aff_sub.columns].apply(lambda col: col.isna().sum(), axis=0)
    missings = missings[missings > n]
    print("Dropping ", missings.shape[0]," Columns")
    print(missings.index)
    aff_sub = aff_sub.drop(missings.index, axis=1)
    return aff_sub, list(missings.index)


def drop_low_cardinality_features(aff_sub, FEATURES):
    feature_cardinality = [len(aff_sub[feature].value_counts()) for feature in FEATURES]
    numerical_features = [feature for feature, cardinality in zip(FEATURES, feature_cardinality) if cardinality > 5]
    categorical_features = [feature for feature, cardinality in zip(FEATURES, feature_cardinality) if (cardinality <= 5) and (cardinality >= 2)]
    no_variance_features = [feature for feature, cardinality in zip(FEATURES, feature_cardinality) if cardinality == 1]
    aff_sub = aff_sub[['roundid', 'sourced?', 'year'] + numerical_features + categorical_features]
    return aff_sub, numerical_features, categorical_features, no_variance_features


def scale_train_and_data_to_score(aff_sub, aff, FEATURES, target_labels):
    '''
    Scales the data to be used for training and scoring.
    :param aff_sub:
    :param aff:
    :param FEATURES:
    :param target_labels:
    :return:
    '''
    # Logistic Regression Anlysis
    # scale numerical features and categorical features are already 0/1.
    temp1 = aff_sub[FEATURES].describe()
    scaler = MinMaxScaler().fit(aff_sub[FEATURES])
    X = scaler.transform(aff_sub[FEATURES])
    X = pd.DataFrame(X, columns=FEATURES).reset_index(drop=True)
    temp2 = X.describe()
    y = aff_sub[target_labels].reset_index(drop=True).values.ravel()
    # Now score each round of AFF from this model that trained on selected data.
    X_to_score = scaler.transform(aff[FEATURES])
    X_to_score = pd.DataFrame(X_to_score, columns=FEATURES).reset_index(drop=True)
    return X, y, X_to_score


def test_model_on_cross_validation(X_train, y_train, model):
    cv = StratifiedKFold(n_splits=4)
    y_scores = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def get_decile_performance(y, probs):
    #y, probs  = y_train, probs_train_rf[:, 1]
    probs_df = pd.DataFrame(np.c_[probs, y], columns=['probs', 'y'])
    deciles = probs_df['probs'].quantile(q=[i / 10 for i in range(0, 11)]).drop_duplicates()
    results = probs_df.groupby(pd.cut(probs_df['probs'], deciles)).apply(
       lambda df: pd.Series({'n': df['y'].count(), 'n sourced': 100 * df['y'].sum() / probs_df['y'].sum()})
    )
    results = results.reset_index()
    results = results.sort_values('probs', ascending=False)
    results['cumulative n sourced'] = results['n sourced'].cumsum()
    return results#.reset_index(drop=True)


def train_and_score(X_train, y_train, X_test, y_test, model_rf, oversample=True, train_size=None):
    # Fill missings with mean
    X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
    X_train = X_train.fillna(-1).values#X_train.median())
    X_test = X_test.fillna(-1).values#X_train.median())
    # Fit model and predict probabilities
    if oversample:
        print('Before SMOTE', X_train.shape, y_train.shape, y_train.mean())
        model_smote = SMOTE(sampling_strategy=.5, random_state=0, k_neighbors=5)
        X_train_ovr, y_train_ovr = model_smote.fit_resample(X_train, y_train)
        print('After SMOTE', X_train_ovr.shape, y_train_ovr.shape, y_train_ovr.mean())
    else:
        X_train_ovr, y_train_ovr = X_train, y_train

    if train_size:
        X_train_ovr, y_train_ovr = X_train_ovr[:train_size], y_train_ovr[:train_size]
        print('After limiting for trainsize', X_train_ovr.shape, y_train_ovr.shape, y_train_ovr.mean())

    model_rf.fit(X_train_ovr, y_train_ovr)
    probs_train_lr = model_rf.predict_proba(X_train)
    probs_test_lr = model_rf.predict_proba(X_test)
    return model_rf, probs_train_lr, probs_test_lr


def evaluate_model(y_actual, y_pred_probs):
    # Evaluate
    #y_actual, y_pred_probs = y_train, probs_train_rf[:, 1]
    decile_dist = get_decile_performance(y_actual, y_pred_probs)

    precision, recall, thresholds = precision_recall_curve(y_actual, y_pred_probs)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = auc(recall, precision)

    perf_metrics = {
        'AUC_ROC': np.round(roc_auc_score(y_actual, y_pred_probs),2),
        'AUC_PR': np.round(auc_precision_recall,2),
        'n_sourced_top2deciles': np.round(decile_dist['n sourced'][0:2].sum(),2)
    }
    return perf_metrics, decile_dist


def run_model_on_4_cv(model_rf, X, y, cv_k, roundids, oversample, train_size):
    train_performance = []
    test_performance = []
    test_predictions = []
    k=4
    for cv_k_i in range(1, k+1):
        #cv_k_i=1
        print(f'CV {cv_k_i}')
        train, test = cv_k[cv_k != cv_k_i].index, cv_k[cv_k == cv_k_i].index
        #print(train.index, test)
        X_train, X_test, y_train, y_test, roundids_test = X[train], X[test], y[train], y[test], roundids[test]
        print('Train Data Summary:', X_train.shape, y_train.shape, y_train.mean())
        print('Test Data Summary:', X_test.shape, y_test.shape, y_test.mean())

        model_rf, probs_train_rf, probs_test_rf = train_and_score(X_train, y_train, X_test, y_test, model_rf, oversample, train_size)

        train_perf, train_decile_dist = evaluate_model(y_train, probs_train_rf[:, 1])
        test_perf, train_decile_dist = evaluate_model(y_test, probs_test_rf[:, 1])
        print("Train Performance", train_perf)
        print("Test Performance", test_perf)

        train_performance.append(pd.Series(train_perf))
        test_performance.append(pd.Series(test_perf))

        #print(roundids[test].shape, probs_test_rf[:, 1].shape)
        test_predictions.append(
            pd.DataFrame({'roundid': roundids_test, 'sourced?': y_test, 'y_pred_prob': probs_test_rf[:, 1]})
        )

    # K fold summary
    cv_perf_summary = pd.concat(train_performance + test_performance,axis=1).T
    cv_perf_summary['train or test'] = np.repeat(['train', 'test'], k)
    print(cv_perf_summary)

    # Overall Test summary
    final_prediced_data = pd.concat(test_predictions).sort_index(axis=0)

    #print(evaluate_model(y, y_probs_test2))
    decile_perf = get_decile_performance(final_prediced_data['sourced?'], final_prediced_data['y_pred_prob'])
    print("Whole Test Set Performance", decile_perf)

    return decile_perf, cv_perf_summary, final_prediced_data
