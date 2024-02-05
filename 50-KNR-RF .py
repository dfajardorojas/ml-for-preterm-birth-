# -*- coding: utf-8 -*-
"""
This file contains all the functions for the implementation of the pipeline
described in the paper Predicting Gestational Age at Birth in the Context
of Preterm Birth From Multi-modal Fetal MRI by Fajardo-Rojas et.al.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import itertools
import time
import seaborn as sns
import warnings
import xgboost as xgb

from collections import Counter
from pylab import *
from time import gmtime, strftime
from datetime import datetime

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn import svm, datasets
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_regression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.utils import resample
from xgboost import XGBRegressor


DATA_FILE_LOCATION = "/path/to/data"
# Replace this with symbol or value that represents missing value in the data. E.g. '-', 0 or -1.0 etc
MISSING_VALUES_PLACEHOLDER = -1
OUTCOME_COL_NAME = "tag_gadel"  # Age at delivery

SHOW_VISUALISATIONS = True
COLORS = [
    "#003f5c",
    "#f95d6a",
    "#6a9e3c",
    "#f2975a",
    "#a05195",
    "#2f4b7c",
    "#665191",
    "#d45087",
    "#ffa600",
]


def load_medical_data(file_name):
    return pd.read_csv(file_name, header="infer")


raw_data = load_medical_data(DATA_FILE_LOCATION)


def replace_missing_values(dataframe):
    return dataframe.replace(MISSING_VALUES_PLACEHOLDER, np.nan)


def remove_rows_without_outcome(dataframe):
    dataframe = dataframe.copy()
    return dataframe.dropna(subset=[OUTCOME_COL_NAME]).reset_index(drop=True)


def remove_cols_with_no_data(dataframe):
    dataframe = dataframe.copy()
    dataframe = dataframe.dropna(axis="columns", how="all")
    return dataframe


def remove_rows_no_3T(dataframe):
    dataframe = dataframe.copy()
    dataframe = dataframe.loc[dataframe["tag_scanner"] == 1]
    return dataframe


def remove_rows_37(dataframe):
    dataframe = dataframe.copy()
    dataframe = dataframe.loc[dataframe['tag_ga'] < 37]
    return dataframe


def divide_to_categories(
    dataframe,
    columns_to_categorize=["tag_gadel"],
    categorized_columns_names=["ga_delivery"],
):
    CATEGORY_TERM = 0  # Term baby GA >= 37
    CATEGORY_PRETERM = 1  # Preterm baby GA < 37

    CATEGORY_LATE_PRETERM = 1  # Late preterm 32 <= GA < 37
    CATEGORY_VERY_PRETERM = 2  # Very preterm 28 <= GA < 32
    CATEGORY_EXTREMELY_PRETERM = 3  # Extremely preterm GA < 28

    dataframe = dataframe.copy()

    for col_id, column_to_categorize in enumerate(columns_to_categorize):
        GA_delivery_array = dataframe[column_to_categorize]
        nr_of_patients = GA_delivery_array.size
        outcome_2_categories = []
        outcome_4_categories = []

        for i in range(0, nr_of_patients):
            if GA_delivery_array[i] < 28:
                outcome_4_categories.append(3)
                outcome_2_categories.append(1)
            elif GA_delivery_array[i] >= 28 and GA_delivery_array[i] < 32:
                outcome_4_categories.append(2)
                outcome_2_categories.append(1)
            elif GA_delivery_array[i] >= 32 and GA_delivery_array[i] < 37:
                outcome_4_categories.append(1)
                outcome_2_categories.append(1)
            else:
                outcome_4_categories.append(0)
                outcome_2_categories.append(0)

        dataframe_4_categories = pd.DataFrame(outcome_4_categories)
        dataframe_4_categories.columns = [
            categorized_columns_names[col_id] + "_4_category"
        ]

        dataframe_2_categories = pd.DataFrame(outcome_2_categories)
        dataframe_2_categories.columns = [
            categorized_columns_names[col_id] + "_binary_category"
        ]

        dataframe = pd.concat(
            [dataframe, dataframe_4_categories, dataframe_2_categories],
            axis=1,
            join="inner",
        )

    return dataframe


def add_35_and_37_category(dataframe):
    outcome_categories_35_and_37 = []

    ga_delivery_column = dataframe["tag_gadel"]
    tech_cohort_column = dataframe["tag_typ"]
    nr_of_patients = ga_delivery_column.size

    for i in range(0, nr_of_patients):
        if ga_delivery_column[i] >= 37 and tech_cohort_column[i] == 99:
            outcome_categories_35_and_37.append(0)
        elif ga_delivery_column[i] <= 35:
            outcome_categories_35_and_37.append(2)
        else:
            outcome_categories_35_and_37.append(1)

    dataframe_35_and_37 = pd.DataFrame(outcome_categories_35_and_37)
    dataframe_35_and_37.columns = ["35_and_37_category"]

    return pd.concat([dataframe, dataframe_35_and_37], axis=1, join="inner")




_data = replace_missing_values(raw_data)
_data = remove_rows_without_outcome(_data)
_data = remove_cols_with_no_data(_data)
_data = divide_to_categories(_data)
_data = add_35_and_37_category(_data)
_data = remove_rows_no_3T(_data)
_data = remove_rows_37(_data)


def get_data():
    return _data.copy()


# Returns only specified columns and drops all rows where at least one of the columns has a NaN value in that row
def get_columns_and_drop_nan_rows(dataframe, list_of_features):
    for feature in list_of_features:
        dataframe = dataframe.dropna(subset=[feature])
    dataframe = dataframe[dataframe.columns.intersection(list_of_features)]
    return dataframe


# Takes in list of column names and converts them to z score column names.
# E.g ['col_1', 'col_2'] -> ['col_1_z_score', 'col_2_z_score']
#The name z-score is used as a shorthand for the method of internally
#studentised residuals
#This function is for features depending of ga at time of mri scan
def convert_to_zscore_column_names(non_zscore_column_names):
    z_names = []
    for col in non_zscore_column_names:
        z_names.append(col + "_z_score")

    return z_names

#This function is for features depending of ga at time of growth US scan
def convert_to_zscore_column_names_gu(non_zscore_column_names):
    z_names = []
    for col in non_zscore_column_names:
        z_names.append(col + '_z_score_gu')

    return z_names


#This function is for features depending of ga at time of anomaly US scan
def convert_to_zscore_column_names_anom(non_zscore_column_names):
    z_names = []
    for col in non_zscore_column_names:
        z_names.append(col + '_z_score_anom')

    return z_names



# Feature names
ORIGINAL_COL_NAMES = get_data().columns
BIRTH_CATEGORIES_4 = ['Term','Late preterm','Very preterm','Extremely preterm']
BIRTH_CATEGORIES_2 = ['Term', 'Preterm']

EXCLUDED_FEATURE_NAMES = ['tag_scanner', 'tag_complete_id']
OUTCOME_FEATURE_NAMES = ['tag_typ', 'tag_gadel', 'tag_mod', 'tag_bwg', 'tag_bwc', 'tag_hc', 'tag_hcc', 'tag_garom', 'tag_histo_weight', 'tag_histo_mvm', 'tag_histo_fvm', 'tag_histo_chorio', 'tag_apgar5', 'tag_del_bwc', 'ga_delivery_4_category', 'ga_delivery_binary_category', 'tag_cat_norm', '35_and_37_category']
NOT_ENOUGH_DATA_FEATURE_NAMES = ['tag_vol_0','tag_vol_1', 'tag_vol_2', 'tag_vol_3', 'tag_vol_4', 'tag_vol_5', 'tag_vol_6', 'tag_vol_7', 'diff_1', 'diff_2', 'diff_3', 'diff_4', 'diff_5', 'diff_6', 'diff_7', 'diff_8', 'diff_9', 'diff_10', 'diff_11', 'lung_t2s_kurt', 'lung_t2s_lacu', 'lung_t2s_mean', 'lung_t2s_skew', 'lung_t2s_vol', 'tag_vol_body', 'tag_anom_cord_ins', 'tag_gu_cord_ins', 'tag_anom_pi_left', 'tag_anom_pi_right', 'tag_cervix_length', 'tag_vol_t2w_complete']

#Threshold of <50% missing data, obtained from get_missing_data_info() function below
ALL_Z_SCORE_FEATURE_NAMES = ['plac_t2s_mean',
                             'plac_t2s_vol',
                             'plac_t2s_lacu',
                             'plac_t2s_skew',
                             'plac_t2s_kurt',
                             'brain_t2s_mean',
                             'brain_t2s_vol',
                             'brain_t2s_lacu',
                             'brain_t2s_skew',
                             'brain_t2s_kurt',
                             'tag_cervix_length',
                             'tag_vol_body',
                             'tag_cptr',
                             'eCSF_L',
                             'eCSF_R',
                             'Cortex_L',
                             'Cortex_R',
                             'WM_L',
                             'WM_R',
                             'Lat_ventricle_L',
                             'Lat_ventricle_R',
                             'CSP',
                             'Brainstem',
                             'Cerebellum_L',
                             'Cerebellum_R',
                             'Vermis',
                             'Lentiform_L',
                             'Lentiform_R',
                             'Thalamus_L',
                             'Thalamus_R',
                             'Third_ventricle',
                             'tag_vol_t2w_complete']


ALL_Z_SCORE_FEATURE_NAMES_GU = ['tag_gu_hc',
                                'tag_gu_ac',
                                'tag_gu_bpd',
                                'tag_gu_fl',
                                'tag_gu_pi',
                                'tag_gu_efw',
                                'tag_gu_edf']

ALL_Z_SCORE_FEATURE_NAMES_ANOM = ['tag_anom_ac',
                             'tag_anom_bpd',
                             'tag_anom_fl',
                             'tag_anom_hc']


ALL_REGULAR_FEATURE_NAMES = ['tag_age',
                             'tag_bmi',
                             'tag_loc',
                             'tag_sex',
                             'tag_parity',
                             'tag_diabetes',
                             'tag_anom_loc',
                             'tag_anom_cord',
                             'tag_ivf',
                             'tag_smok',
                             'tag_prev_ptb',
                             'tag_gu_efw_cen',
                             'tag_gu_loc',
                             'tag_bp_sys',
                             'tag_bp_dias',
                             'tag_bp_hr']
# ALL_FEATURE_NAMES =  convert_to_zscore_column_names(ALL_Z_SCORE_FEATURE_NAMES) + ALL_REGULAR_FEATURE_NAMES
ALL_FEATURE_NAMES =  convert_to_zscore_column_names(ALL_Z_SCORE_FEATURE_NAMES) + convert_to_zscore_column_names_gu(ALL_Z_SCORE_FEATURE_NAMES_GU) + convert_to_zscore_column_names_anom(ALL_Z_SCORE_FEATURE_NAMES_ANOM) + ALL_REGULAR_FEATURE_NAMES


epsilon = 2.22045e-16
#https://onlinelibrary.wiley.com/doi/10.7863/ultra.16.03025
def add_column_with_z_score(dataframe, column_name, show_graphs):
    "Computation of the method of internally studentised residuals"
    "for GA at time of MRI"
    dataframe = dataframe.copy()
    ga_at_scan_key = 'tag_ga'
    ga_at_delivery_key = 'tag_gadel'
    tech_cohort_key = 'tag_typ'

    if ga_at_scan_key not in dataframe:
        print("Dataframe is missing 'tag_ga'(age at scan)")
        return

    # Returns dataframe with Age at scan and specified measurement / feature
    relevant_data = get_columns_and_drop_nan_rows(dataframe, [ga_at_scan_key, ga_at_delivery_key, column_name, tech_cohort_key])[[ga_at_scan_key, ga_at_delivery_key, column_name, tech_cohort_key]]

    # Only take the controls to create regression lines
    # relevant_data = relevant_data.loc[(relevant_data[ga_at_delivery_key] >= 37) & (relevant_data[tech_cohort_key] == 99)]

    X = relevant_data[ga_at_scan_key].to_numpy().reshape(-1, 1)
    y = relevant_data[column_name]


    if show_graphs:
        # Plot scatterplot of the specific column vs age at scan
        fig, axs = plt.subplots(2, figsize = (20, 20))
        fig.suptitle('Regression lines for {} z-score'.format(column_name))
        axs[0].plot(X, y, '.m', label = '{}'.format(column_name))
        axs[0].set_xlabel('{}'.format(ga_at_scan_key))
        axs[0].set_ylabel('{}'.format(column_name))
        axs[0].legend()
        fig.tight_layout()

        axs[0].set_axisbelow(True)
        axs[0].yaxis.grid(color='gray', linestyle='solid')
        axs[0].xaxis.grid(color='gray', linestyle='solid')


    # Perform linear regression
    linear_regressor_means = LinearRegression()
    linear_regressor_means.fit(X, y)
    y_pred_means = linear_regressor_means.predict(X)
    coefs_means = linear_regressor_means.coef_[0]
    intercept_means = linear_regressor_means.intercept_

    if show_graphs:
        # Plot linear regression line for the mean
        y_line_function = coefs_means*X + intercept_means
        axs[0].plot(X, y_line_function, '-k')
        print('[{}] Linear regression line for the mean: y = {}*X + ({})'.format(column_name, coefs_means, intercept_means))

    residuals = relevant_data[column_name] - y_pred_means
    # scaled_residuals = residuals * math.sqrt(math.pi / 2)
    # abs_scaled_residuals = scaled_residuals.abs()

    if show_graphs:
        # Plot scatterplot of the residuals vs age at scan
        axs[1].plot(X, residuals, '.g', label = 'residuals for term babies')
        axs[1].set_xlabel('{}'.format(ga_at_scan_key))
        axs[1].set_ylabel('{}'.format(column_name))
        axs[1].legend()
        fig.tight_layout()

    residuals_squared = residuals**2
    length = len(residuals_squared) - 1
    std = np.sqrt(np.sum(residuals_squared/length))
    #Diagonal of Hat Matrix
    only_diag = np.einsum('ij, ij -> j', X.T, np.linalg.pinv(X))
    j=0

    for i, row in dataframe.iterrows():
        ga_at_scan = row[ga_at_scan_key]
        if (np.isnan(row[column_name])) or np.isnan(ga_at_scan):
            continue
        measurement = row[column_name]
        #We use the mean of 'tag_ga' column as a rough estimate of missing values
        if np.isnan(ga_at_scan):
          ga_at_scan = dataframe[ga_at_scan_key].mean()
        z_score = (measurement - linear_regressor_means.predict([[ga_at_scan]])) / ((std*np.sqrt(1-only_diag[j]))+epsilon)
        dataframe.loc[i, column_name + '_z_score'] = z_score

    return dataframe


def add_columns_with_z_score(dataframe, columns, show_graphs = False):
    for col in columns:
        dataframe = add_column_with_z_score(dataframe, col, show_graphs)

    return dataframe


def add_column_with_z_score_gu(dataframe, column_name, show_graphs):
    "Computation of the method of internally studentised residuals"
    "for GA at time of growth US"
    dataframe = dataframe.copy()
    ga_at_scan_key = 'tag_gu_ga'
    ga_at_delivery_key = 'tag_gadel'
    tech_cohort_key = 'tag_typ'

    if ga_at_scan_key not in dataframe:
        print("Dataframe is missing 'tag_gu_ga'(age at scan)")
        return

    # Returns dataframe with Age at scan and specified measurement / feature
    relevant_data = get_columns_and_drop_nan_rows(dataframe, [ga_at_scan_key, ga_at_delivery_key, column_name, tech_cohort_key])[[ga_at_scan_key, ga_at_delivery_key, column_name, tech_cohort_key]]

    # Only take the controls to create regression lines
    # relevant_data = relevant_data.loc[(relevant_data[ga_at_delivery_key] >= 37) & (relevant_data[tech_cohort_key] == 99)]

    X = relevant_data[ga_at_scan_key].to_numpy().reshape(-1, 1)
    y = relevant_data[column_name]


    if show_graphs:
        # Plot scatterplot of the specific column vs age at scan
        fig, axs = plt.subplots(2, figsize = (20, 20))
        fig.suptitle('Regression lines for {} z-score_gu'.format(column_name))
        axs[0].plot(X, y, '.m', label = '{} for term babies'.format(column_name))
        axs[0].set_xlabel('{}'.format(ga_at_scan_key))
        axs[0].set_ylabel('{}'.format(column_name))
        axs[0].legend()
        fig.tight_layout()

        axs[0].set_axisbelow(True)
        axs[0].yaxis.grid(color='gray', linestyle='solid')
        axs[0].xaxis.grid(color='gray', linestyle='solid')


    # Perform linear regression
    linear_regressor_means = LinearRegression()
    linear_regressor_means.fit(X, y)
    y_pred_means = linear_regressor_means.predict(X)
    coefs_means = linear_regressor_means.coef_[0]
    intercept_means = linear_regressor_means.intercept_

    if show_graphs:
        # Plot linear regression line for the mean
        y_line_function = coefs_means*X + intercept_means
        axs[0].plot(X, y_line_function, '-k')
        print('[{}] Linear regression line for the mean: y = {}*X + ({})'.format(column_name, coefs_means, intercept_means))

    residuals = relevant_data[column_name] - y_pred_means
    # scaled_residuals = residuals * math.sqrt(math.pi / 2)
    # abs_scaled_residuals = scaled_residuals.abs()

    if show_graphs:
        # Plot scatterplot of the residuals vs age at scan
        axs[1].plot(X, residuals, '.g', label = 'residuals for term babies')
        axs[1].set_xlabel('{}'.format(ga_at_scan_key))
        axs[1].set_ylabel('{}'.format(column_name))
        axs[1].legend()
        fig.tight_layout()


    residuals_squared = residuals**2
    length = len(residuals_squared) - 1
    std = np.sqrt(np.sum(residuals_squared/length))
    #Diagonal of Hat Matrix
    only_diag = np.einsum('ij, ij -> j', X.T, np.linalg.pinv(X))
    j=0

    for i, row in dataframe.iterrows():
        ga_at_scan = row[ga_at_scan_key]
        if (np.isnan(row[column_name])) or np.isnan(ga_at_scan):
            continue
        measurement = row[column_name]
        #We use the mean of 'tag_ga' column as a rough estimate of missing values
        if np.isnan(ga_at_scan):
          ga_at_scan = dataframe[ga_at_scan_key].mean()
        z_score = (measurement - linear_regressor_means.predict([[ga_at_scan]])) / ((std*np.sqrt(1-only_diag[j]))+epsilon)
        dataframe.loc[i, column_name + '_z_score_gu'] = z_score

    return dataframe


def add_columns_with_z_score_gu(dataframe, columns, show_graphs = False):
    for col in columns:
        dataframe = add_column_with_z_score_gu(dataframe, col, show_graphs)

    return dataframe


def add_column_with_z_score_anom(dataframe, column_name, show_graphs):
    "Computation of the method of internally studentised residuals"
    "for GA at time of anomaly US"
    dataframe = dataframe.copy()
    ga_at_scan_key = 'tag_anom_ga'
    ga_at_delivery_key = 'tag_gadel'
    tech_cohort_key = 'tag_typ'

    if ga_at_scan_key not in dataframe:
        print("Dataframe is missing 'tag_anom_ga'(age at scan)")
        return

    # Returns dataframe with Age at scan and specified measurement / feature
    relevant_data = get_columns_and_drop_nan_rows(dataframe, [ga_at_scan_key, ga_at_delivery_key, column_name, tech_cohort_key])[[ga_at_scan_key, ga_at_delivery_key, column_name, tech_cohort_key]]

    # Only take the controls to create regression lines
    # relevant_data = relevant_data.loc[(relevant_data[ga_at_delivery_key] >= 37) & (relevant_data[tech_cohort_key] == 99)]

    X = relevant_data[ga_at_scan_key].to_numpy().reshape(-1, 1)
    y = relevant_data[column_name]


    if show_graphs:
        # Plot scatterplot of the specific column vs age at scan
        fig, axs = plt.subplots(2, figsize = (20, 20))
        fig.suptitle('Regression lines for {} z-score_gu'.format(column_name))
        axs[0].plot(X, y, '.m', label = '{} for term babies'.format(column_name))
        axs[0].set_xlabel('{}'.format(ga_at_scan_key))
        axs[0].set_ylabel('{}'.format(column_name))
        axs[0].legend()
        fig.tight_layout()

        axs[0].set_axisbelow(True)
        axs[0].yaxis.grid(color='gray', linestyle='solid')
        axs[0].xaxis.grid(color='gray', linestyle='solid')


    # Perform linear regression
    linear_regressor_means = LinearRegression()
    linear_regressor_means.fit(X, y)
    y_pred_means = linear_regressor_means.predict(X)
    coefs_means = linear_regressor_means.coef_[0]
    intercept_means = linear_regressor_means.intercept_

    if show_graphs:
        # Plot linear regression line for the mean
        y_line_function = coefs_means*X + intercept_means
        axs[0].plot(X, y_line_function, '-k')
        print('[{}] Linear regression line for the mean: y = {}*X + ({})'.format(column_name, coefs_means, intercept_means))

    residuals = relevant_data[column_name] - y_pred_means
    # scaled_residuals = residuals * math.sqrt(math.pi / 2)
    # abs_scaled_residuals = scaled_residuals.abs()

    if show_graphs:
        # Plot scatterplot of the residuals vs age at scan
        axs[1].plot(X, residuals, '.g', label = 'residuals for term babies')
        axs[1].set_xlabel('{}'.format(ga_at_scan_key))
        axs[1].set_ylabel('{}'.format(column_name))
        axs[1].legend()
        fig.tight_layout()


    residuals_squared = residuals**2
    length = len(residuals_squared) - 1
    std = np.sqrt(np.sum(residuals_squared/length))
    #Diagonal of Hat Matrix
    only_diag = np.einsum('ij, ij -> j', X.T, np.linalg.pinv(X))
    j=0

    for i, row in dataframe.iterrows():
        ga_at_scan = row[ga_at_scan_key]
        if (np.isnan(row[column_name])) or np.isnan(ga_at_scan):
            continue
        measurement = row[column_name]
        #We use the mean of 'tag_ga' column as a rough estimate of missing values
        if np.isnan(ga_at_scan):
          ga_at_scan = dataframe[ga_at_scan_key].mean()
        z_score = (measurement - linear_regressor_means.predict([[ga_at_scan]])) / ((std*np.sqrt(1-only_diag[j]))+epsilon)
        dataframe.loc[i, column_name + '_z_score_anom'] = z_score

    return dataframe


def add_columns_with_z_score_anom(dataframe, columns, show_graphs = False):
    for col in columns:
        dataframe = add_column_with_z_score_anom(dataframe, col, show_graphs)

    return dataframe



def perform_imputation(dataframe):
    "Function to perform imputation of dataframe"

    columns = dataframe.columns
    all_data = dataframe.values
    # Get indices of columns that should be taken for imputation

    if "plac_t2s_kurt_z_score" in columns:
        plac_t2s_kurt_z_score_index = columns.get_loc("plac_t2s_kurt_z_score")
    else:
        plac_t2s_kurt_z_score_index = -1
        print('plac_t2s_kurt_z_score NOT AVAILABLE')

    if "plac_t2s_lacu_z_score" in columns:
        plac_t2s_lacu_z_score_index = columns.get_loc("plac_t2s_lacu_z_score")
    else:
        plac_t2s_lacu_z_score_index = -1
        print('plac_t2s_lacu_z_score NOT AVAILABLE')

    if "plac_t2s_mean_z_score" in columns:
        plac_t2s_mean_z_score_index = columns.get_loc("plac_t2s_mean_z_score")
    else:
        plac_t2s_mean_z_score_index = -1
        print('plac_t2s_mean_z_score NOT AVAILABLE')

    if "plac_t2s_skew_z_score" in columns:
        plac_t2s_skew_z_score_index = columns.get_loc("plac_t2s_skew_z_score")
    else:
        plac_t2s_skew_z_score_index = -1
        print('plac_t2s_skew_z_score NOT AVAILABLE')

    if "plac_t2s_vol_z_score" in columns:
        plac_t2s_vol_z_score_index = columns.get_loc("plac_t2s_vol_z_score")
    else:
        plac_t2s_vol_z_score_index = -1
        print('plac_t2s_vol_z_score NOT AVAILABLE')

    if "brain_t2s_kurt_z_score" in columns:
        brain_t2s_kurt_z_score_index = columns.get_loc("brain_t2s_kurt_z_score")
    else:
        brain_t2s_kurt_z_score_index = -1
        print('brain_t2s_kurt_z_score NOT AVAILABLE')

    if "brain_t2s_lacu_z_score" in columns:
        brain_t2s_lacu_z_score_index = columns.get_loc("brain_t2s_lacu_z_score")
    else:
        brain_t2s_lacu_z_score_index = -1
        print('brain_t2s_lacu_z_score NOT AVAILABLE')

    if "brain_t2s_mean_z_score" in columns:
        brain_t2s_mean_z_score_index = columns.get_loc("brain_t2s_mean_z_score")
    else:
        brain_t2s_mean_z_score_index = -1
        print('brain_t2s_mean_z_score NOT AVAILABLE')

    if "brain_t2s_skew_z_score" in columns:
        brain_t2s_skew_z_score_index = columns.get_loc("brain_t2s_skew_z_score")
    else:
        brain_t2s_skew_z_score_index = -1
        print('brain_t2s_skew_z_score NOT AVAILABLE')

    if "brain_t2s_vol_z_score" in columns:
        brain_t2s_vol_z_score_index = columns.get_loc("brain_t2s_vol_z_score")
    else:
        brain_t2s_vol_z_score_index = -1
        print('brain_t2s_vol_z_score NOT AVAILABLE')

    if "tag_cervix_length_z_score" in columns:
        tag_cervix_length_z_score_index = columns.get_loc("tag_cervix_length_z_score")
    else:
        tag_cervix_length_z_score_index = -1
        print('tag_cervix_length_z_score NOT AVAILABLE')

    if "tag_vol_body_z_score" in columns:
        tag_vol_body_z_score_index = columns.get_loc("tag_vol_body_z_score")
    else:
        tag_vol_body_z_score_index = -1
        print('tag_vol_body_z_score NOT AVAILABLE')

    if "tag_cptr_z_score" in columns:
        tag_cptr_z_score_index = columns.get_loc("tag_cptr_z_score")
    else:
        tag_cptr_z_score_index = -1
        print('tag_cptr_z_score NOT AVAILABLE')

    if "eCSF_L_z_score" in columns:
        eCSF_L_z_score_index = columns.get_loc("eCSF_L_z_score")
    else:
        eCSF_L_z_score_index = -1
        print('eCSF_L_z_score NOT AVAILABLE')

    if "eCSF_R_z_score" in columns:
        eCSF_R_z_score_index = columns.get_loc("eCSF_R_z_score")
    else:
        eCSF_R_z_score_index = -1
        print('eCSF_R_z_score NOT AVAILABLE')

    if "Cortex_L_z_score" in columns:
        Cortex_L_z_score_index = columns.get_loc("Cortex_L_z_score")
    else:
        Cortex_L_z_score_index = -1
        print('Cortex_L_z_score NOT AVAILABLE')

    if "Cortex_R_z_score" in columns:
        Cortex_R_z_score_index = columns.get_loc("Cortex_R_z_score")
    else:
        Cortex_R_z_score_index = -1
        print('Cortex_R_z_score NOT AVAILABLE')

    if "WM_L_z_score" in columns:
        WM_L_z_score_index = columns.get_loc("WM_L_z_score")
    else:
        WM_L_z_score_index = -1
        print('WM_L_z_score NOT AVAILABLE')

    if "WM_R_z_score" in columns:
        WM_R_z_score_index = columns.get_loc("WM_R_z_score")
    else:
        WM_R_z_score_index = -1
        print('WM_R_z_score NOT AVAILABLE')

    if "Lat_ventricle_L_z_score" in columns:
        Lat_ventricle_L_z_score_index = columns.get_loc("Lat_ventricle_L_z_score")
    else:
        Lat_ventricle_L_z_score_index = -1
        print('Lat_ventricle_L_z_score NOT AVAILABLE')

    if "Lat_ventricle_R_z_score" in columns:
        Lat_ventricle_R_z_score_index = columns.get_loc("Lat_ventricle_R_z_score")
    else:
        Lat_ventricle_R_z_score_index = -1
        print('Lat_ventricle_R_z_score NOT AVAILABLE')

    if "CSP_z_score" in columns:
        CSP_z_score_index = columns.get_loc("CSP_z_score")
    else:
        CSP_z_score_index = -1
        print('CSP_z_score NOT AVAILABLE')

    if "Brainstem_z_score" in columns:
        Brainstem_z_score_index = columns.get_loc("Brainstem_z_score")
    else:
        Brainstem_z_score_index = -1
        print('Brainstem_z_score NOT AVAILABLE')

    if "Cerebellum_L_z_score" in columns:
        Cerebellum_L_z_score_index = columns.get_loc("Cerebellum_L_z_score")
    else:
        Cerebellum_L_z_score_index = -1
        print('Cerebellum_L_z_score NOT AVAILABLE')

    if "Cerebellum_R_z_score" in columns:
        Cerebellum_R_z_score_index = columns.get_loc("Cerebellum_R_z_score")
    else:
        Cerebellum_R_z_score_index = -1
        print('Cerebellum_R_z_score NOT AVAILABLE')

    if "Vermis_z_score" in columns:
        Vermis_z_score_index = columns.get_loc("Vermis_z_score")
    else:
        Vermis_z_score_index = -1
        print('Vermis_z_score NOT AVAILABLE')

    if "Lentiform_L_z_score" in columns:
        Lentiform_L_z_score_index = columns.get_loc("Lentiform_L_z_score")
    else:
        Lentiform_L_z_score_index = -1
        print('Lentiform_L_z_score NOT AVAILABLE')

    if "Lentiform_R_z_score" in columns:
        Lentiform_R_z_score_index = columns.get_loc("Lentiform_R_z_score")
    else:
        Lentiform_R_z_score_index = -1
        print('Lentiform_R_z_score NOT AVAILABLE')

    if "Thalamus_L_z_score" in columns:
        Thalamus_L_z_score_index = columns.get_loc("Thalamus_L_z_score")
    else:
        Thalamus_L_z_score_index = -1
        print('Thalamus_L_z_score NOT AVAILABLE')

    if "Thalamus_R_z_score" in columns:
        Thalamus_R_z_score_index = columns.get_loc("Thalamus_R_z_score")
    else:
        Thalamus_R_z_score_index = -1
        print('Thalamus_R_z_score NOT AVAILABLE')

    if "Third_ventricle_z_score" in columns:
        Third_ventricle_z_score_index = columns.get_loc("Third_ventricle_z_score")
    else:
        Third_ventricle_z_score_index = -1
        print('Third_ventricle_z_score NOT AVAILABLE')

    if "tag_vol_t2w_complete_z_score" in columns:
        tag_vol_t2w_complete_z_score_index = columns.get_loc("tag_vol_t2w_complete_z_score")
    else:
        tag_vol_t2w_complete_z_score_index = -1
        print('tag_vol_t2w_complete_z_score NOT AVAILABLE')

    if "tag_gu_ac_z_score_gu" in columns:
        tag_gu_ac_z_score_gu_index = columns.get_loc("tag_gu_ac_z_score_gu")
    else:
        tag_gu_ac_z_score_gu_index = -1
        print('tag_gu_ac_z_score_gu NOT AVAILABLE')

    if "tag_gu_bpd_z_score_gu" in columns:
        tag_gu_bpd_z_score_gu_index = columns.get_loc("tag_gu_bpd_z_score_gu")
    else:
        tag_gu_bpd_z_score_gu_index = -1
        print('tag_gu_bpd_z_score_gu NOT AVAILABLE')

    if "tag_gu_fl_z_score_gu" in columns:
        tag_gu_fl_z_score_gu_index = columns.get_loc("tag_gu_fl_z_score_gu")
    else:
        tag_gu_fl_z_score_gu_index = -1
        print('tag_gu_fl_z_score_gu NOT AVAILABLE')

    if "tag_gu_hc_z_score_gu" in columns:
        tag_gu_hc_z_score_gu_index = columns.get_loc("tag_gu_hc_z_score_gu")
    else:
        tag_gu_hc_z_score_gu_index = -1
        print('tag_gu_hc_z_score_gu NOT AVAILABLE')

    if "tag_gu_pi_z_score_gu" in columns:
        tag_gu_pi_z_score_gu_index = columns.get_loc("tag_gu_pi_z_score_gu")
    else:
        tag_gu_pi_z_score_gu_index = -1
        print('tag_gu_pi_z_score_gu NOT AVAILABLE')

    if "tag_gu_efw_z_score_gu" in columns:
        tag_gu_efw_z_score_gu_index = columns.get_loc("tag_gu_efw_z_score_gu")
    else:
        tag_gu_efw_z_score_gu_index = -1
        print('tag_gu_efw_z_score_gu NOT AVAILABLE')

    if "tag_gu_edf_z_score_gu" in columns:
        tag_gu_edf_z_score_gu_index = columns.get_loc("tag_gu_edf_z_score_gu")
    else:
        tag_gu_edf_z_score_gu_index = -1
        print('tag_gu_edf_z_score_gu NOT AVAILABLE')

    if "tag_age" in columns:
        tag_age_index = columns.get_loc("tag_age")
    else:
        tag_age_index = -1
        print('tag_age NOT AVAILABLE')

    if "tag_bmi" in columns:
        tag_bmi_index = columns.get_loc("tag_bmi")
    else:
        tag_bmi_index = -1
        print('tag_bmi NOT AVAILABLE')

    if "tag_loc" in columns:
        tag_loc_index = columns.get_loc("tag_loc")
    else:
        tag_loc_index = -1
        print('tag_loc NOT AVAILABLE')

    if "tag_sex" in columns:
        tag_sex_index = columns.get_loc("tag_sex")
    else:
        tag_sex_index = -1
        print('tag_sex NOT AVAILABLE')

    if "tag_parity" in columns:
        tag_parity_index = columns.get_loc("tag_parity")
    else:
        tag_parity_index = -1
        print('tag_parity NOT AVAILABLE')

    if "tag_diabetes" in columns:
        tag_diabetes_index = columns.get_loc("tag_diabetes")
    else:
        tag_diabetes_index = -1
        print('tag_diabetes NOT AVAILABLE')

    if "tag_anom_loc" in columns:
        tag_anom_loc_index = columns.get_loc("tag_anom_loc")
    else:
        tag_anom_loc_index = -1
        print('tag_anom_loc NOT AVAILABLE')

    if "tag_anom_cord" in columns:
        tag_anom_cord_index = columns.get_loc("tag_anom_cord")
    else:
        tag_anom_cord_index = -1
        print('tag_anom_cord NOT AVAILABLE')

    if "tag_ivf" in columns:
        tag_ivf_index = columns.get_loc("tag_ivf")
    else:
        tag_ivf_index = -1
        print('tag_ivf NOT AVAILABLE')

    if "tag_smok" in columns:
        tag_smok_index = columns.get_loc("tag_smok")
    else:
        tag_smok_index = -1
        print('tag_smok NOT AVAILABLE')

    if "tag_prev_ptb" in columns:
        tag_prev_ptb_index = columns.get_loc("tag_prev_ptb")
    else:
        tag_prev_ptb_index = -1
        print('tag_prev_ptb NOT AVAILABLE')

    if "tag_gu_efw_cen" in columns:
        tag_gu_efw_cen_index = columns.get_loc("tag_gu_efw_cen")
    else:
        tag_gu_efw_cen_index = -1
        print('tag_gu_efw_cen NOT AVAILABLE')

    if "tag_gu_loc" in columns:
        tag_gu_loc_index = columns.get_loc("tag_gu_loc")
    else:
        tag_gu_loc_index = -1
        print('tag_gu_loc NOT AVAILABLE')

    if "tag_bp_sys" in columns:
        tag_bp_sys_index = columns.get_loc("tag_bp_sys")
    else:
        tag_bp_sys_index = -1
        print('tag_bp_sys NOT AVAILABLE')

    if "tag_bp_dias" in columns:
        tag_bp_dias_index = columns.get_loc("tag_bp_dias")
    else:
        tag_bp_dias_index = -1
        print('tag_bp_dias NOT AVAILABLE')

    if "tag_bp_hr" in columns:
        tag_bp_hr_index = columns.get_loc("tag_bp_hr")
    else:
        tag_bp_hr_index = -1
        print('tag_bp_hr NOT AVAILABLE')

    if "tag_anom_ac_z_score_anom" in columns:
        tag_anom_ac_z_score_anom_index = columns.get_loc("tag_anom_ac_z_score_anom")
    else:
        tag_anom_ac_z_score_anom_index = -1
        print('tag_anom_ac_z_score_anom NOT AVAILABLE')

    if "tag_anom_bpd_z_score_anom" in columns:
        tag_anom_bpd_z_score_anom_index = columns.get_loc("tag_anom_bpd_z_score_anom")
    else:
        tag_anom_bpd_z_score_anom_index = -1
        print('tag_anom_bpd_z_score_anom NOT AVAILABLE')

    if "tag_anom_fl_z_score_anom" in columns:
        tag_anom_fl_z_score_anom_index = columns.get_loc("tag_anom_fl_z_score_anom")
    else:
        tag_anom_fl_z_score_anom_index = -1
        print('tag_anom_fl_z_score_anom NOT AVAILABLE')


    if "tag_anom_hc_z_score_anom" in columns:
        tag_anom_hc_z_score_anom_index = columns.get_loc("tag_anom_hc_z_score_anom")
    else:
        tag_anom_hc_z_score_anom_index = -1
        print('tag_anom_hc_z_score_anom NOT AVAILABLE')


    #THESE ONES WON'T BE IMPUTED BUT ADDED LATER

    if "tag_complete_id" in columns:
        tag_complete_id_index = columns.get_loc("tag_complete_id")
    else:
        tag_complete_id_index = -1
        print('tag_complete_id NOT AVAILABLE')

    if "ga_delivery_4_category" in columns:
        ga_delivery_4_category_index = columns.get_loc("ga_delivery_4_category")
    else:
        ga_delivery_4_category_index = -1
        print('ga_delivery_4_category NOT AVAILABLE')

    if "ga_delivery_binary_category" in columns:
        ga_delivery_binary_category_index = columns.get_loc("ga_delivery_binary_category")
    else:
        ga_delivery_binary_category_index = -1
        print('ga_delivery_binary_category NOT AVAILABLE')

    if "35_and_37_category" in columns:
        help_35_and_37_category_index = columns.get_loc("35_and_37_category")
    else:
        help_35_and_37_category_index = -1
        print('35_and_37_category NOT AVAILABLE')

    if "tag_gadel" in columns:
        tag_gadel_index = columns.get_loc("tag_gadel")


    # Array of all columns except for GA at delivery and the columns with no meaning (id of patients etc)
    ix = [i for i in range(all_data.shape[1]) if (i == plac_t2s_kurt_z_score_index or
                                                  i == plac_t2s_lacu_z_score_index or
                                                  i == plac_t2s_mean_z_score_index or
                                                  i == plac_t2s_skew_z_score_index or
                                                  i == plac_t2s_vol_z_score_index or
                                                  i == brain_t2s_kurt_z_score_index or
                                                  i == brain_t2s_lacu_z_score_index or
                                                  i == brain_t2s_mean_z_score_index or
                                                  i == brain_t2s_skew_z_score_index or
                                                  i == brain_t2s_vol_z_score_index or
                                                  i == tag_cervix_length_z_score_index or
                                                  i == tag_vol_body_z_score_index or
                                                  i == tag_cptr_z_score_index or
                                                  i == eCSF_L_z_score_index or
                                                  i == eCSF_R_z_score_index or
                                                  i == Cortex_L_z_score_index or
                                                  i == Cortex_R_z_score_index or
                                                  i == WM_L_z_score_index or
                                                  i == WM_R_z_score_index or
                                                  i == Lat_ventricle_L_z_score_index or
                                                  i == Lat_ventricle_R_z_score_index or
                                                  i == CSP_z_score_index or
                                                  i == Brainstem_z_score_index or
                                                  i == Cerebellum_L_z_score_index or
                                                  i == Cerebellum_R_z_score_index or
                                                  i == Vermis_z_score_index or
                                                  i == Lentiform_L_z_score_index or
                                                  i == Lentiform_R_z_score_index or
                                                  i == Thalamus_L_z_score_index or
                                                  i == Thalamus_R_z_score_index or
                                                  i == Third_ventricle_z_score_index or
                                                  i == tag_vol_t2w_complete_z_score_index or
                                                  i == tag_gu_ac_z_score_gu_index or
                                                  i == tag_gu_bpd_z_score_gu_index or
                                                  i == tag_gu_fl_z_score_gu_index or
                                                  i == tag_gu_hc_z_score_gu_index or
                                                  i == tag_gu_pi_z_score_gu_index or
                                                  i == tag_gu_efw_z_score_gu_index or
                                                  i == tag_gu_edf_z_score_gu_index or
                                                  i == tag_age_index or
                                                  i == tag_bmi_index or
                                                  i == tag_loc_index or
                                                  i == tag_sex_index or
                                                  i == tag_parity_index or
                                                  i == tag_diabetes_index or
                                                  i == tag_anom_loc_index or
                                                  i == tag_anom_cord_index or
                                                  i == tag_ivf_index or
                                                  i == tag_smok_index or
                                                  i == tag_prev_ptb_index or
                                                  i == tag_gu_efw_cen_index or
                                                  i == tag_gu_loc_index or
                                                  i == tag_bp_sys_index or
                                                  i == tag_bp_dias_index or
                                                  i == tag_bp_hr_index or
                                                  i == tag_anom_ac_z_score_anom_index or
                                                  i == tag_anom_bpd_z_score_anom_index or
                                                  i == tag_anom_fl_z_score_anom_index or
                                                  i == tag_anom_hc_z_score_anom_index)]

    X = all_data[:, ix] # Features are all rows of relevant columns
    y = all_data[:, tag_gadel_index] # Labels are GA at delivery

    # define imputer, can be changed accordingly for other experiments
    imputer = IterativeImputer(estimator=KNeighborsRegressor(weights='distance'),
                               initial_strategy='mean',
                               max_iter=10, random_state=1,
                               verbose=11)
    # fit on the dataset
    Xtrans = imputer.fit_transform(X)

    df_imputation = pd.DataFrame(Xtrans)

    # Take only column names from indices which were used for continuous imputation
    col_names_for_imputation = columns[ix]

    # Assign the correct column name for the imputated cols
    df_imputation.columns = col_names_for_imputation

    #Round categorical values to nearest integer
    df_imputation = df_imputation.round({"tag_loc": 0,
                                         "tag_sex": 0,
                                         "tag_parity": 0,
                                         "tag_prev_ptb": 0,
                                         "tag_diabetes": 0,
                                         "tag_anom_loc": 0,
                                         "tag_anom_cord": 0,
                                         "tag_smok": 0,
                                         "tag_ivf": 0,
                                         "tag_gu_loc": 0})

    # Insert back important columns that were not imputated

    if "tag_gadel" in columns:
        df_imputation.insert(loc = 0, column = columns[tag_gadel_index], value = y)

    if "tag_complete_id" in columns:
        df_imputation.insert(loc = 0, column = columns[tag_complete_id_index], value = all_data[:, tag_complete_id_index])

    if "ga_delivery_4_category" in columns:
        df_imputation.insert(loc = 0, column = columns[ga_delivery_4_category_index], value = all_data[:, ga_delivery_4_category_index])

    if "ga_delivery_binary_category" in columns:
        df_imputation.insert(loc = 0, column = columns[ga_delivery_binary_category_index], value = all_data[:, ga_delivery_binary_category_index])

    if "35_and_37_category" in columns:
        df_imputation.insert(loc = 0, column = columns[help_35_and_37_category_index], value = all_data[:, help_35_and_37_category_index])


    return df_imputation


_data_with_z_scores = add_columns_with_z_score(get_data(), ALL_Z_SCORE_FEATURE_NAMES, False)

#To verify that the z_scores columns are added instead of replacing things
#display(_data_with_z_scores.to_string())

def get_data_with_z_scores():
    return _data_with_z_scores.copy()

_data_with_z_scores_gu = add_columns_with_z_score_gu(get_data_with_z_scores(), ALL_Z_SCORE_FEATURE_NAMES_GU, False)

def get_data_with_z_scores_gu():
    return _data_with_z_scores_gu.copy()

_data_with_z_scores_anom = add_columns_with_z_score_anom(get_data_with_z_scores_gu(), ALL_Z_SCORE_FEATURE_NAMES_ANOM, False)

def get_data_with_z_scores_anom():
    return _data_with_z_scores_anom.copy()

def get_processed_data():
    data = get_data()
    data = add_columns_with_z_score(data, ALL_Z_SCORE_FEATURE_NAMES)
    data = add_columns_with_z_score_gu(data, ALL_Z_SCORE_FEATURE_NAMES_GU)
    data = add_columns_with_z_score_anom(data, ALL_Z_SCORE_FEATURE_NAMES_ANOM)
    data = perform_imputation(data)

    return data.copy()



prueba = get_processed_data()


def get_missing_data_info(print_missing=False):
    dataframe = get_data()
    col_names_with_values = []
    nr_of_rows = dataframe.shape[0]

    missing_values_info = {}

    for i in range(dataframe.shape[1]):
        array = dataframe.iloc[:, i].values
        nr_of_NaN = np.count_nonzero(np.isnan(array))
        col_names_with_values.append(dataframe.columns[i])

        # Find percentage of missing values
        percentage_of_missing_vals = round((nr_of_NaN / nr_of_rows) * 100, 2)

        missing_values_info[dataframe.columns[i]] = percentage_of_missing_vals
        # missing_values_info[dataframe.columns[i] + '_z_score'] = percentage_of_missing_vals

        # Count number of rows with missing values
        if print_missing:
            print(
                "> %d, %s, Missing: %d (%.1f%%)"
                % (i, dataframe.columns[i], nr_of_NaN, percentage_of_missing_vals)
            )

    return missing_values_info


def get_feature_importances(data):
    # generate dataset
    X = data[ALL_FEATURE_NAMES]
    y = data['tag_gadel'].ravel()
    # define feature selection model, can be changed accordingly for other experiments
    fs = SelectFromModel(RandomForestRegressor(random_state=1))
    # apply feature selection
    #X_selected = fs.fit_transform(X, y)
    fs.fit_transform(X, y)
    cols = fs.get_support(indices=True)
    features_df_new = X.iloc[:,cols]

    feature_with_scores = {}

    for i, feature in enumerate(ALL_FEATURE_NAMES):
        feature_with_scores[feature] = fs.estimator_.feature_importances_[i]

    return {k: v for k, v in sorted(feature_with_scores.items(), key=lambda item: item[1], reverse=True)}


def get_k_best_labels_features(data, k):
    return [i[0] for i in list(feature_importances.items())[:k]]


def get_k_best_feature_labels_by_importances(feature_importances, k):
    return [i[0] for i in list(feature_importances.items())[:k]]



feature_importances = get_feature_importances(prueba)
selected_feature_labels = get_k_best_feature_labels_by_importances(
    feature_importances, 10
)
selected_features = prueba[selected_feature_labels]





def get_svr_model():
    model = Pipeline([("svr", SVR(max_iter=1000000))])
    parameters = {'svr__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'svr__gamma':['scale', 'auto'],
                  'svr__kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                  'svr__epsilon': [0.001 ,0.01 ,0.1, 0.5, 1],
                  'svr__degree': [2, 3]}
    return (model, parameters, 'SVR model')

def get_rf_model():
    model = Pipeline([("rf", RandomForestRegressor(n_jobs=-1, random_state=1))])
    parameters = {'rf__max_depth': [3, 5, 10, 20, 50, 100],
                  'rf__max_features': ['auto', 'sqrt', 'log2'],
                  'rf__n_estimators': [5, 10, 20, 50, 100, 250]}
    return (model, parameters, 'RF model')

def get_xg_model():
    model = Pipeline([("xg", XGBRegressor(n_jobs=-1, random_state=1))])
    parameters = {"xg__learning_rate": [0.01, 0.05, 0.10, 0.3, 0.5],
                  "xg__max_depth": [ 3, 5, 7, 10, 20, 50],
                  "xg__min_child_weight": [ 1, 3, 5, 7],
                  "xg__gamma":[ 0.1, 0.5, 0.8, 2, 5, 10],
                  "xg__colsample_bytree":[0.3, 0.5, 0.7]}
    return (model, parameters, 'XG model')


def get_stats_per_model(X, labels, model_pipeline, model_parameters, stratification_type, random_state, cv_folds):
    "Function to train a single model and validate"
    X = X.to_numpy()
    X_shape_two_tenths = 2*int(np.ceil(X.shape[0]/10))
    if (stratification_type is not False):
        X_train, X_val_test, y_train, y_val_test, strat_train, strat_val_test = train_test_split(X, labels, stratification_type, test_size = X_shape_two_tenths, stratify=stratification_type, random_state = random_state )
    else:
        X_train, X_val_test, y_train, y_val_test = train_test_split(X, labels, test_size = X_shape_two_tenths, random_state = random_state )

    y_train_binary = np.copy(y_train)

    y_train_binary[y_train_binary < 37] = 1
    y_train_binary[y_train_binary >= 37] = 0


    X_train_preterm = X_train[y_train_binary==1, :]
    y_train_preterm = y_train[y_train_binary==1]

    X_train_term = X_train[y_train_binary==0, :]
    y_train_term = y_train[y_train_binary==0]

    #Upsampling step
    X_train_preterm_resam, y_train_preterm_resam  = resample(X_train_preterm, y_train_preterm, n_samples=X_train_term.shape[0], random_state=1)

    X_train_upsampled = np.concatenate((X_train_term, X_train_preterm_resam), axis=0)
    y_train_upsampled = np.concatenate((y_train_term, y_train_preterm_resam), axis=0)
    y_train_upsampled = np.reshape(y_train_upsampled, (-1, 1))

    X_and_y_train_upsampled = np.concatenate((X_train_upsampled, y_train_upsampled), axis=1)


    np.random.seed(1)
    np.random.shuffle(X_and_y_train_upsampled)

    X_train_upsampled = X_and_y_train_upsampled[:, :-1]
    y_train_upsampled = X_and_y_train_upsampled[:, -1]


    if (stratification_type is not False):
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5, stratify=strat_val_test, random_state = random_state )
    else:
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5, random_state = random_state )

    grid_search = GridSearchCV(model_pipeline, model_parameters, cv = cv_folds, refit = 'r2', scoring = ['r2', 'neg_mean_absolute_error'], verbose = 11, n_jobs=-1)

    scaler = StandardScaler()

    #weights = X_train[:, -1]
    #weights = weights.copy(order='C')

    #X_train = X_train[:, 0:-1]
    #X_test = X_test[:, 0:-1]
    X_train = scaler.fit_transform(X_train)
    X_train_upsampled = scaler.transform(X_train_upsampled)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    #print('X_train shape: ', X_train.shape)
    #print('X_val shape: ', X_val.shape)
    #print('X_test shape: ', X_test.shape)




    grid_search.fit(X_train_upsampled, y_train_upsampled)
    best_model_params = grid_search.best_params_
    keys_model_params = list(best_model_params.keys())

    #Get index of best R2 score
    dict_of_results = grid_search.cv_results_
    #print(dict_of_results.keys())
    dict_of_indices = dict()
    for key in keys_model_params:
      list_of_possible_values = dict_of_results["param_"+ key]
      dict_of_indices[key] = set(np.where(np.array(list_of_possible_values) == best_model_params[key])[0])
    list_of_sets_of_possible_indices = list(dict_of_indices.values())
    (index, ) = set.intersection(*list_of_sets_of_possible_indices)

    r2_train_cv = dict_of_results['mean_test_r2'][index]
    mean_error_train_cv = dict_of_results['mean_test_neg_mean_absolute_error'][index]

    y_pred_train = grid_search.predict(X_train)

    y_pred_val = grid_search.predict(X_val)

    y_true_val = y_val.reshape(-1)
    r2_val = r2_score(y_true_val, y_pred_val)

    mean_error_val = mean_absolute_error(y_true_val, y_pred_val)

    y_pred_test = grid_search.predict(X_test)

    y_true_test = y_test.reshape(-1)
    r2_test = r2_score(y_true_test, y_pred_test)

    mean_error_test = mean_absolute_error(y_true_test, y_pred_test)

    return r2_val, mean_error_val, r2_test, mean_error_test, r2_train_cv, mean_error_train_cv, best_model_params, y_pred_train, y_pred_val, y_pred_test, y_train, y_true_val, y_true_test

output_file_path = "/output/path/"

# Stratification
labels_binary_classification = get_data()['ga_delivery_binary_category'].to_numpy()
labels_4_cat = get_data()['ga_delivery_4_category'].to_numpy()
labels_35_37_cat = get_data()['35_and_37_category'].to_numpy()

warnings.filterwarnings('ignore')

#Manually checked that binary strat does better
stratification_variations = [(labels_binary_classification, 'binary_cat')]


def try_different_combinations(feature_labels,
                               nr_of_features_to_try,
                               list_number_features_for_meta,
                               meta_lr = False,
                               meta_rf = False,
                               ):

    "Function to train and validate different models, as well as stacking"
    "with a linear regression or random forest meta-model."
    "This function takes as input the list of features selected by the feature"
    "selection step, as well as a list of the number of features to give as input"
    "to the base models."

    # Create the time-stamped file for when the function is run
    time_str = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    f = open(output_file_path + '/resultsML_'+ time_str +'.tsv', 'w+')

    models = [get_rf_model(), get_svr_model(), get_xg_model()] # Add models to be tested

    data = get_processed_data()
    labels = data['tag_gadel'].values.reshape(-1, 1)
    labels = labels.ravel()


    headers = ['Model', 'Parameters', 'Combination', 'Stratified', 'CV folds', 'Random state', 'R-score Val', 'Mean error Val', 'R-score Test', 'Mean error Test', 'Acc Test', 'Sen Test', 'Spe Test', 'R-score TrainCV', 'Mean error TrainCV']
    header_row = '\t'.join(headers) + '\n'
    f.write(header_row)



    predictions_matrix_train = np.zeros((np.shape(data)[0]-2*int(np.ceil(data.shape[0]/10)), len(stratification_variations)*len(models)*(2**(len(nr_of_features_to_try))-1)))
    predictions_matrix_val = np.zeros((int(np.ceil(data.shape[0]/10)), len(stratification_variations)*len(models)*(2**(len(nr_of_features_to_try))-1)))
    predictions_matrix_test = np.zeros((int(np.ceil(data.shape[0]/10)), len(stratification_variations)*len(models)*(2**(len(nr_of_features_to_try))-1)))
    #print('Shape train: ', predictions_matrix_train.shape)
    #print('Shape val: ', predictions_matrix_val.shape)

    current_column = 0
    best_r2_val = 0


    #Make an empty list for each number of features to try
    #The first element is a list of values
    #The second element a list of indices
    dict_lists_for_each_number_features = dict()
    for numb_feat in list_number_features_for_meta:
        dict_lists_for_each_number_features[str(numb_feat)] = [[],[]]

    for (stratification, stratification_description) in stratification_variations:
        for cv_folds in [3]:
            for random_state in [1]:
                for model in models:
                    for nr_of_features in nr_of_features_to_try:
                        combinations = list(itertools.combinations(feature_labels, nr_of_features))
                        for i, combination in enumerate(combinations):
                            r2_val, y_pred_train, y_pred_val, y_pred_test, y_train, y_true_val, y_true_test = execute_calculation(data,
                                                                                                                          combination,
                                                                                                                          labels,
                                                                                                                          model[0],
                                                                                                                          model[1],
                                                                                                                          model[2],
                                                                                                                          stratification,
                                                                                                                          stratification_description,
                                                                                                                          random_state,
                                                                                                                          cv_folds,
                                                                                                                          #weights,
                                                                                                                         f)
                            for numb_feat in list_number_features_for_meta:
                                if len(dict_lists_for_each_number_features[str(numb_feat)][1]) < numb_feat:
                                    dict_lists_for_each_number_features[str(numb_feat)][0].append(r2_val)
                                    dict_lists_for_each_number_features[str(numb_feat)][1].append(current_column)
                                elif len(dict_lists_for_each_number_features[str(numb_feat)][1]) == numb_feat:
                                    if r2_val > min(dict_lists_for_each_number_features[str(numb_feat)][0]):
                                        ind_min = dict_lists_for_each_number_features[str(numb_feat)][0].index(min(dict_lists_for_each_number_features[str(numb_feat)][0]))
                                        dict_lists_for_each_number_features[str(numb_feat)][0][ind_min] = r2_val
                                        dict_lists_for_each_number_features[str(numb_feat)][1][ind_min] = current_column




                            if (meta_lr or meta_rf) == True:
                              predictions_matrix_train[:, current_column] = y_pred_train
                              predictions_matrix_test[:, current_column] = y_pred_test
                              predictions_matrix_val[:, current_column] = y_pred_val
                              current_column += 1

                            #print(y_pred_train.shape)
                            #print(y_pred_val.shape)
    #print(predictions_matrix_train)

    if meta_lr == True:

      for numb_feat in list_number_features_for_meta:

        X_meta_train = predictions_matrix_train[:, dict_lists_for_each_number_features[str(numb_feat)][1]]
        X_meta_val = predictions_matrix_val[:, dict_lists_for_each_number_features[str(numb_feat)][1]]
        X_meta_test = predictions_matrix_test[:, dict_lists_for_each_number_features[str(numb_feat)][1]]

        #print('linear', numb_feat, 'X_meta_train shape', X_meta_train.shape)
        #print('linear', numb_feat, 'X_meta_val shape', X_meta_val.shape)
        #print('linear', numb_feat, 'X_meta_test shape', X_meta_test.shape)

        scaler = StandardScaler()
        X_meta_train = scaler.fit_transform(X_meta_train)
        X_meta_val = scaler.transform(X_meta_val)
        X_meta_test = scaler.transform(X_meta_test)
        meta_linear_regressor = LinearRegression()
        meta_linear_regressor.fit(X_meta_train, y_train)
        y_pred_meta_val = meta_linear_regressor.predict(X_meta_val)
        y_pred_meta_test = meta_linear_regressor.predict(X_meta_test)
        y_pred_meta_train = meta_linear_regressor.predict(X_meta_train)
        r2_train = r2_score(y_train, y_pred_meta_train)
        mean_error_train = mean_absolute_error(y_train, y_pred_meta_train)
        r2_val = r2_score(y_true_val, y_pred_meta_val)
        mean_error_val = mean_absolute_error(y_true_val, y_pred_meta_val)
        r2_test = r2_score(y_true_test, y_pred_meta_test)
        mean_error_test = mean_absolute_error(y_true_test, y_pred_meta_test)

        if r2_val > best_r2_val:
            best_y_pred_meta_val = y_pred_meta_val
            best_y_pred_meta_test = y_pred_meta_test
            best_r2_val = r2_val

        result = '\t'.join(['META LINEAR REGRESSOR', '.', str(numb_feat), '.', '.', str(random_state), str(r2_val), str(mean_error_val), str(r2_test), str(mean_error_test), str(r2_train), str(mean_error_train)])
        f.write(result + '\n')
        f.flush()



    if meta_rf == True:

      for numb_feat in list_number_features_for_meta:

        X_meta_train = predictions_matrix_train[:, dict_lists_for_each_number_features[str(numb_feat)][1]]
        X_meta_val = predictions_matrix_val[:, dict_lists_for_each_number_features[str(numb_feat)][1]]
        X_meta_test = predictions_matrix_test[:, dict_lists_for_each_number_features[str(numb_feat)][1]]

        #print('rf', numb_feat, 'X_meta_train shape', X_meta_train.shape)
        #print('rf', numb_feat, 'X_meta_val shape', X_meta_val.shape)
        #print('rf', numb_feat, 'X_meta_test shape', X_meta_test.shape)

        scaler = StandardScaler()
        X_meta_train = scaler.fit_transform(X_meta_train)
        X_meta_val = scaler.transform(X_meta_val)
        X_meta_test = scaler.transform(X_meta_test)
        model = get_rf_model()
        model_meta_rf = model[0]
        model_parameters = model[1]

        grid_search_meta_rf = GridSearchCV(model_meta_rf,
                                           model_parameters,
                                           cv = cv_folds,
                                           refit = 'r2',
                                           scoring = ['r2', 'neg_mean_absolute_error'],
                                           #verbose = 11,
                                           n_jobs=-1)
        grid_search_meta_rf.fit(X_meta_train, y_train)
        best_model_params = grid_search_meta_rf.best_params_
        keys_model_params = list(best_model_params.keys())

        #Get index of best R2 score
        dict_of_results = grid_search_meta_rf.cv_results_
        dict_of_indices = dict()
        for key in keys_model_params:
          list_of_possible_values = dict_of_results["param_"+ key]
          dict_of_indices[key] = set(np.where(np.array(list_of_possible_values) == best_model_params[key])[0])
        list_of_sets_of_possible_indices = list(dict_of_indices.values())
        (index, ) = set.intersection(*list_of_sets_of_possible_indices)

        r2_train_cv = dict_of_results['mean_test_r2'][index]
        mean_error_train_cv = dict_of_results['mean_test_neg_mean_absolute_error'][index]

        y_pred_meta_val = grid_search_meta_rf.predict(X_meta_val)
        y_pred_meta_test = grid_search_meta_rf.predict(X_meta_test)
        r2_val = r2_score(y_true_val, y_pred_meta_val)
        mean_error_val = mean_absolute_error(y_true_val, y_pred_meta_val)
        r2_test = r2_score(y_true_test, y_pred_meta_test)
        mean_error_test = mean_absolute_error(y_true_test, y_pred_meta_test)

        if r2_val > best_r2_val:
            best_y_pred_meta_val = y_pred_meta_val
            best_y_pred_meta_test = y_pred_meta_test
            best_r2_val = r2_val


        result = '\t'.join(['META RF REGRESSOR', str(best_model_params), str(numb_feat), '.', '.', str(random_state), str(r2_val), str(mean_error_val), str(r2_test), str(mean_error_test), '.', '.', '.', str(r2_train_cv), str(mean_error_train_cv)])
        f.write(result + '\n')
        f.flush()





    print('Finished!')
    return best_y_pred_meta_val, best_y_pred_meta_test



def execute_calculation(data,
                        combination,
                        labels,
                        model,
                        model_parameters,
                        model_description,
                        stratification,
                        stratification_description,
                        random_state,
                        cv_folds,
                        f):

    "Function to evaluate trained base models on test set"

    features_comb = data[list(combination)]

    (r2_val, mean_error_val, r2_test, mean_error_test, r2_train_cv, mean_error_train_cv, best_model_params, y_pred_train, y_pred_val, y_pred_test, y_train, y_true_val, y_true_test) = get_stats_per_model(
        features_comb,
        labels,
        model,
        model_parameters,
        stratification,
        random_state,
        cv_folds)


    y_true_test_binary = np.copy(y_true_test)
    y_pred_test_binary = np.copy(y_pred_test)

    y_true_test_binary[y_true_test_binary < 37] = 1
    y_true_test_binary[y_true_test_binary >= 37] = 0

    y_pred_test_binary[y_pred_test_binary < 37] = 1
    y_pred_test_binary[y_pred_test_binary >= 37] = 0


    acc_test = accuracy_score(y_true_test_binary, y_pred_test_binary)
    sen_test = recall_score(y_true_test_binary, y_pred_test_binary)
    spe_test = recall_score(y_true_test_binary, y_pred_test_binary, pos_label = 0)

    acc_test = round(acc_test, 2)
    sen_test = round(sen_test, 2)
    spe_test = round(spe_test, 2)



    result = '\t'.join([str(model_description), str(best_model_params), str(combination), str(stratification_description), str(cv_folds), str(random_state), str(r2_val), str(mean_error_val), str(r2_test), str(mean_error_test), str(acc_test), str(sen_test), str(spe_test), str(r2_train_cv), str(mean_error_train_cv)])

    f.write(result + '\n')
    f.flush()
    return r2_val, y_pred_train, y_pred_val, y_pred_test, y_train, y_true_val, y_true_test


#Main training function, with the features selected by the feature selection step
best_y_pred_meta_val, best_y_pred_meta_test = try_different_combinations(
    (
        'tag_cervix_length_z_score',
        'plac_t2s_mean_z_score',
        'tag_gu_edf_z_score_gu',
        'tag_cptr_z_score',
        'tag_anom_bpd_z_score_anom',
        'plac_t2s_kurt_z_score',
        'tag_anom_hc_z_score_anom',
        'brain_t2s_kurt_z_score',
        'tag_gu_efw_z_score_gu',
        'brain_t2s_vol_z_score'
    ),
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [203, 101, 14],
    True,
    True
)


def save_fig(fileName, plt):
    plt.savefig('/path/{}.jpg'.format(fileName), bbox_inches='tight', dpi=250)


def plot_data_identity(x_data, y_data, categories_data, category_labels, xlabel, ylabel,graph_title, plot_line_of_best_fit, identity):
    "Function to plot predictions of model"

    plt.rcParams.update({'font.size': 30})
    fig, ax = plt.subplots(figsize=(15,15))
    lines_of_best_fit_labels = []
    legend = []

    # Marker configuration
    m = ['.','.', '.', '.', '.', 'd', '^', 'X']
    markercolor = COLORS
    marker_size = 17

    for k, category_label in enumerate(category_labels):
        # Pick the right values
        x_vals = x_data[categories_data==k]
        y_vals = y_data[categories_data==k]
        # Remove vals where y-value is NaN and make x and y the same size
        x_vals = x_vals[~np.isnan(y_vals)]
        y_vals = y_vals[~np.isnan(y_vals)]
        plt.plot(x_vals, y_vals, m[k], alpha=1, markerfacecolor = markercolor[k], markersize = marker_size, markeredgecolor = 'k', markeredgewidth = 0.5)
        legend.append(category_label)
        if plot_line_of_best_fit:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            plt.plot(x_vals, slope*x_vals + intercept, linestyle='solid', color=markercolor[k], linewidth = 3)
            print('slope:', slope, category_label)
            print('intercept', intercept, category_label)
            legend.append(category_label + ' line of best fit')


    plt.legend(legend, prop={'size': 25}, loc = 'lower right')



    #Add identity line
    if identity:
      plt.plot( [20,42],[20,42], linewidth = 3)
      legend.append('identity line')



    plt.title('Diagnosis of preterm birth')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(graph_title)



    major_ticks = [20,22,24,26,28,30,32,34,37,40,42]

    # Add grid behind plot
    ax.grid()
    ax.axis('equal')
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which='major', linestyle=':', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    a = ax.get_ygridlines()
    b = a[8]
    b.set_color('red')
    b.set_linewidth(3)
    c = a[6]
    c.set_color('green')
    c.set_linewidth(3)
    d = a[4]
    d.set_color('orange')
    d.set_linewidth(3)


    save_fig(graph_title, plt)


def evaluation(y, y_pred):

    # accuracy
    acc = accuracy_score(y, y_pred)
    print('accuracy: ', round(acc,2))

    # default is sensitivity: pos_label = 1
    sensitivity = recall_score(y,y_pred)
    print('sensitivity: ',round(sensitivity,2))
    # pos_label = 0 gives specificity
    specificity = recall_score(y,y_pred,pos_label = 0)
    print('specificity: ',round(specificity,2))







def try_from_best_meta_r2(best_y_pred_meta_val, best_y_pred_meta_test):
    "Function to visualise results of best meta-model"
    data_to_use = get_processed_data()

    y = data_to_use['tag_gadel'].values.reshape(-1, 1)
    y = y.ravel()

    #scaler = StandardScaler()


    X_shape_two_tenths = 2*int(np.ceil(data_to_use.shape[0]/10))

    stratification_type = labels_binary_classification

    if (stratification_type is not False):
        X_train, X_val_test, y_train, y_val_test, strat_train, strat_val_test = train_test_split(data_to_use, y, stratification_type, test_size = X_shape_two_tenths, stratify=stratification_type, random_state = 1)
    else:
        X_train, X_val_test, y_train, y_val_test = train_test_split(data_to_use, y, test_size = X_shape_two_tenths, random_state = 1)

    #print('X_train shape: ', X_train.shape)
    #print('X_val_test shape: ', X_val_test.shape)

    if (stratification_type is not False):
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5, stratify=strat_val_test, random_state = 1)
    else:
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5, random_state = 1)




    y_true_val = y_val.reshape(-1)
    r2_val = r2_score(y_true_val, best_y_pred_meta_val)

    mean_error_val = mean_absolute_error(y_true_val, best_y_pred_meta_val)


    y_true_test = y_test.reshape(-1)
    r2_test = r2_score(y_true_test, best_y_pred_meta_test)

    mean_error_test = mean_absolute_error(y_true_test, best_y_pred_meta_test)


    print('R2 val is: {}'.format(r2_val))
    print('MEAN ABS val is: {}'.format(mean_error_val))

    print('R2 test is: {}'.format(r2_test))
    print('MEAN ABS test is: {}'.format(mean_error_test))

    y_true_val_binary = np.copy(y_true_val)
    y_pred_val_binary = np.copy(best_y_pred_meta_val)
    y_true_test_binary = np.copy(y_true_test)
    y_pred_test_binary = np.copy(best_y_pred_meta_test)

    y_true_val_binary[y_true_val_binary < 37] = 1
    y_true_val_binary[y_true_val_binary >= 37] = 0

    y_pred_val_binary[y_pred_val_binary < 37] = 1
    y_pred_val_binary[y_pred_val_binary >= 37] = 0

    y_true_test_binary[y_true_test_binary < 37] = 1
    y_true_test_binary[y_true_test_binary >= 37] = 0

    y_pred_test_binary[y_pred_test_binary < 37] = 1
    y_pred_test_binary[y_pred_test_binary >= 37] = 0



    evaluation(y_true_test_binary, y_pred_test_binary)

    evaluation(y_true_val_binary, y_pred_val_binary)



    results_df_test = pd.DataFrame({'real_values': y_true_test, 'predicted_values': best_y_pred_meta_test})

    svr_df_test = divide_to_categories(results_df_test, ['real_values', 'predicted_values'], ['real_values', 'predicted_values'])

    if (SHOW_VISUALISATIONS):
        plot_data_identity(y_true_test, best_y_pred_meta_test, svr_df_test['real_values_4_category'], BIRTH_CATEGORIES_4, 'True GA at birth (weeks)', 'Predicted GA at birth (weeks)', '50-KNR-RF', False, True)



    results_df_val = pd.DataFrame({'real_values': y_true_val, 'predicted_values': best_y_pred_meta_val})

    svr_df_val = divide_to_categories(results_df_val, ['real_values', 'predicted_values'], ['real_values', 'predicted_values'])

    if (SHOW_VISUALISATIONS):
        plot_data_identity(y_true_val, best_y_pred_meta_val, svr_df_val['real_values_4_category'], BIRTH_CATEGORIES_4, 'True GA at birth (weeks)', 'Predicted GA at birth (weeks)', 'Threshold 50%, KNR-RF, Validation Set', False, True)

    return svr_df_test, svr_df_val


try_from_best_meta_r2(best_y_pred_meta_val, best_y_pred_meta_test)
