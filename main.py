"""DS Home Task"""
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.inspection import PartialDependenceDisplay, partial_dependence


def load_data(data_base_path: str):
    """
    This function open the CSV file in pandas framework.
    :param data_base_path: path to data.
    :return: lists of dataframes (for each CSV file)
    """
    csv_files = [os.path.join(data_base_path, f) for f in os.listdir(data_base_path)]
    assert all([os.path.exists(f) for f in csv_files]), "Not al files exists"
    dataframes = []
    for idx, csv in enumerate(csv_files):
        df = pd.read_csv(csv)
        df = df.loc[:, df.columns.str.find('.') < 0]
        df.columns = df.columns.str.strip()
        df['SourceFile'] = idx  # so i concat the CSV files without losing attack type information
        dataframes.append(df)
    return dataframes


def check_if_can_concat(dfs: list, label_column: str):
    """
    This function checks if the different CSV files can be concatenated into one
    by checking if they have the same features and same data types.
    :param dfs: List of DataFrames to be concatenated
    :param label_column: Name of the label column
    :return: True if the DataFrames can be concatenated, False otherwise
    """
    # Extract features and data types for each DataFrame
    features_info = [(df.drop(columns=label_column).columns.tolist(), df.drop(columns=label_column).dtypes) for df in dfs]

    # Check if all features are the same across DataFrames
    if not all(features == features_info[0][0] for features, _ in features_info):
        # Not all CSV files contain the same features
        return False

    # Check if all data types are the same across DataFrames
    if not all(dtypes.equals(features_info[0][1]) for _, dtypes in features_info):
        # Not all CSV files contain the same data types
        return False

    return True


def split_data_to_features_and_target(df, label_column):
    """
    This function split the dataset into features and target classes for prepare it to train.
    :param df: dataset in pandas framework
    :param label_column: str - name of label column
    :return:
    """
    features = list(df.columns)
    features.remove(label_column)
    # This feature add to data information about attack type (because of the way the CSV file created).
    # So I want to remove it because I will not get this information in the real world.
    features.remove('SourceFile')
    return df[features], df[label_column]


def target_to_numeric(df):
    """
    This function perform preprocess on the labels to convert them to numerical.
    :param df: pandas dataframe
    :return: df - the updated dataframe, mapping_classes- mapping between the number and the labels.
    """
    label_encoder = LabelEncoder()
    df = label_encoder.fit_transform(df)
    mapping_classes = label_encoder.classes_
    return df, mapping_classes


def features_preprocessing(df):
    """
    this function cleans the data:
    identify any missing values, normalize features, Convert non-numeric values to numeric values
    :param df: pandas dataframe
    :return: df - the updated dataframe
    """
    features = list(df.columns)
    for feature_name in features:
        if df[feature_name].isna().any():
            df.loc[:, feature_name] = df[feature_name].interpolate(method='linear')
        if np.isinf(df[feature_name]).any():
            max_val = df.loc[~np.isinf(df[feature_name]), feature_name].max()
            df.loc[np.isinf(df[feature_name]), feature_name] = max_val
        assert not np.isneginf(df[feature_name]).any(), "there are -inf values"
        assert not np.isinf(df[feature_name]).any(), "there are inf values"
        assert not df[feature_name].isna().any(), "there are NaN values"
    return df[features]


def data_visualization(df):
    # Plot histograms for each feature, splitting into multiple plots
    num_cols = len(df.columns)
    num_plots = num_cols // 8 + 1
    for i in range(num_plots):
        start_idx = i * 8
        end_idx = min((i + 1) * 8, num_cols)
        cols_to_plot = df.columns[start_idx:end_idx]

        plt.figure(figsize=(20, 8))
        for j, col in enumerate(cols_to_plot):
            plt.subplot(2, 4, j + 1)
            plt.hist(df[col], bins=50, edgecolor='black', alpha=0.7)
            plt.title(col)
            plt.xlabel('')
            plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    # Plot correlation heatmap
    plt.figure(figsize=(16, 10))
    plt.matshow(df.corr(), cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.title('Correlation Heatmap')
    plt.show()


def classes_histogram(df):
    counter_results = Counter(df)
    values = list(counter_results.keys())
    frequencies = list(counter_results.values())
    plt.bar(mapping_classes[values], frequencies, alpha=0.5)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Counter Results')
    plt.xticks(rotation=45)
    plt.show()


def remove_low_variance_features(df):
    threshold = 0.1
    low_variance_features = df.columns[df.var() < threshold]
    return df.drop(columns=low_variance_features)


def reduce_features_with_high_correlation(df):
    corr_matrix = df.corr().abs()
    # Create a mask to identify highly correlated features
    high_corr_mask = (corr_matrix >= 0.8) & (corr_matrix < 1.0)
    features_to_remove = []
    # Iterate through columns and identify highly correlated features
    for col in high_corr_mask.columns:
        correlated_cols = high_corr_mask[col][high_corr_mask[col]].index.tolist()
        for correlated_col in correlated_cols:
            if correlated_col != col and correlated_col not in features_to_remove:
                features_to_remove.append(correlated_col)
    return df.drop(columns=features_to_remove)


def normalize_features(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled


def train_classifier(X_train, X_test, y_train):
    clf = RandomForestClassifier(class_weight='balanced', n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred


def model_evaluations(df_features, X_train, y_test, clf, y_pred):
    """
    This function evaluate the model by calculates the accuracy, precision, recall, confusion matrix,

    :param df_features:
    :param X_train:
    :param y_test:
    :param clf:
    :param y_pred:
    :return:
    """
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    overall_precision = precision_score(y_test, y_pred, average='macro')
    overall_recall = recall_score(y_test, y_pred, average='macro')
    print("Precision: ", overall_precision)
    print("Recall", overall_recall)

    cm = confusion_matrix(y_test, y_pred)

    def get_min_and_max_f1_classes(y_test, y_pred):
        class_report = metrics.classification_report(y_test, y_pred, target_names=mapping_classes, output_dict=True)
        # Extract F1-scores for each class
        f1_scores = {class_name: report['f1-score'] for class_name, report in class_report.items() if
                     class_name != 'accuracy'}
        # Find the class with the minimum F1-score
        min_class = min(f1_scores, key=f1_scores.get)
        max_class = max(f1_scores, key=f1_scores.get)
        print("The class with minimum f1-score is: ", min_class)
        print("The class with maximum f1-score is: ", max_class)

    def plot_features_importance(X_train, clf):
        feature_names = X_train.columns
        feature_importances = clf.feature_importances_
        # Sort feature importances in descending order
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_feature_importances = feature_importances[sorted_indices]
        sorted_feature_names = feature_names[sorted_indices]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_feature_importances)), sorted_feature_importances, align='center')
        plt.yticks(range(len(sorted_feature_importances)), sorted_feature_names)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Random Forest Feature Importance')
        plt.show()

    def create_ppd(clf, X_train, df_features):
        features, feature_names = [(0,)], list(df_features.columns)
        deciles = {0: np.linspace(0, 1, num=5)}
        pd_results = partial_dependence(clf, X_train, features=[0], kind="average", grid_resolution=5,
                                        feature_names=feature_names)
        display = PartialDependenceDisplay([pd_results], features=features, feature_names=feature_names,
                                           target_idx=2, deciles=deciles)
        display.plot()

    get_min_and_max_f1_classes(y_test, y_pred)
    plot_features_importance(X_train, clf)
    create_ppd(clf, X_train, df_features)


if __name__ == '__main__':
    data_base_path = r'data_files/'
    dataframes = load_data(data_base_path)
    label_column = 'Label'
    check_if_can_concat(dataframes, label_column)
    # after I ensure the CSV files are the same, concat them into one dataframe
    combined_df = pd.concat(dataframes, ignore_index=True)
    df_features, df_label = split_data_to_features_and_target(combined_df, label_column)
    df_features = features_preprocessing(df_features)
    df_label, mapping_classes = target_to_numeric(df_label)

    # classes_histogram(df_label)
    # data_visualization(df_features)

    # By visualized the data I noticed that there are some features that all the values are 0 so I decided to remove
    # them and all features with low variance (less than 0.1) and reduce feature with high correlation
    df_features = remove_low_variance_features(df_features)
    df_features = reduce_features_with_high_correlation(df_features)
    df_normalized_features = normalize_features(df_features)

    X_train, X_test, y_train, y_test = train_test_split(df_normalized_features, df_label, test_size=0.30)
    clf, y_pred = train_classifier(X_train, X_test, y_train)
    model_evaluations(df_features, X_train, y_test, clf, y_pred)

    exit(0)
