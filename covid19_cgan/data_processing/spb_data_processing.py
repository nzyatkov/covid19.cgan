import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator

from data_processing.data_processing import DataProcessing
from logger.logger import get_logger


class SpbDataProcessing(DataProcessing):
    mm_scaler = MinMaxScaler()
    data = None
    ml_data = None
    ml_data_transformed = None
    residual_features = None

    @staticmethod
    def _prepare_rnn_dataset(X_residual_features, X_features_to_normalize, basis_Y, features_to_normalize, look_back):
        datares = []
        for start in range(X_residual_features.shape[0] - look_back + 1):
            res_part = X_residual_features[start:start + look_back]

            if features_to_normalize != []:
                norm_part = X_features_to_normalize[start:start + look_back]
                norm_part = pd.DataFrame(data=MinMaxScaler().fit_transform(norm_part),
                                         index=norm_part.index,
                                         columns=norm_part.columns)

                res_part = res_part.join(norm_part)

            datares.append(res_part.values)

        X = np.atleast_3d(np.array(datares))
        X = pd.Series(data=list(X),
                      index=X_residual_features.index[look_back - 1:])

        y = basis_Y[look_back - 1:]

        y = pd.Series(data=list(y.values),
                      index=y.index)

        return X, y

    @staticmethod
    def prepare_spb_rnn_dataset(covid_ml_data_file,
                                selected_features, features_to_normalize, target_vars, target_vars_flatten,
                                start_train, nsample, nsample_forward):
        # Load Data
        data = pd.read_csv(covid_ml_data_file, index_col="Date", parse_dates=True, na_values=['nan'])

        # Preparing data for training
        residual_features = [item for item in selected_features if item not in features_to_normalize]
        ml_data = data[selected_features + target_vars_flatten]
        ml_data = ml_data.loc[start_train:]
        ml_data = ml_data[:-nsample_forward]

        # Data normalization
        ml_data_transformed = pd.DataFrame(
            data=SpbDataProcessing.mm_scaler.fit_transform(ml_data[residual_features + target_vars_flatten]),
            index=ml_data.index,
            columns=ml_data[residual_features + target_vars_flatten].columns)
        ml_data_transformed = ml_data_transformed.join(ml_data[features_to_normalize])

        # Preparing input and output data pairs for a recurrent network
        X, y = SpbDataProcessing._prepare_rnn_dataset(ml_data_transformed[residual_features],
                                                      ml_data_transformed[features_to_normalize],
                                                      ml_data_transformed[target_vars[0]],
                                                      features_to_normalize,
                                                      nsample)

        SpbDataProcessing.data = data
        SpbDataProcessing.ml_data_transformed = ml_data_transformed
        SpbDataProcessing.residual_features = residual_features
        SpbDataProcessing.ml_data = ml_data

        return X, y

    @staticmethod
    def pred_inverse_transform(targets, target_vars, all_y, all_y_predicted, NTraj):
        y_pred_transformed = [None] * len(targets)

        for i in range(len(targets)):
            y_pred_transformed[i] = {}

            col = 'true'
            ml_data_transformed_copy = (SpbDataProcessing
                                        .ml_data_transformed[SpbDataProcessing.residual_features]
                                        .loc[all_y[i].index[0]:all_y[i].index[-1]])
            for j in range(len(targets)):
                ml_data_transformed_copy[target_vars[j]] = all_y[j].values
            y_pred_transformed[i][col] = pd.DataFrame(data=SpbDataProcessing.mm_scaler.inverse_transform(ml_data_transformed_copy),
                                                      index=ml_data_transformed_copy.index,
                                                      columns=ml_data_transformed_copy.columns)[target_vars[i]]

            col = 'pred'
            y_pred_transformed[i][col] = []
            for ntraj in range(NTraj):
                ml_data_transformed_copy = (SpbDataProcessing
                                            .ml_data_transformed[SpbDataProcessing.residual_features]
                                            .loc[all_y[i].index[0]:all_y[i].index[-1]])
                for j in range(len(targets)):
                    ml_data_transformed_copy[target_vars[j]] = all_y_predicted[j][ntraj]
                inv_tr_res_df = pd.DataFrame(data=SpbDataProcessing.mm_scaler.inverse_transform(ml_data_transformed_copy),
                                             index=ml_data_transformed_copy.index,
                                             columns=ml_data_transformed_copy.columns)[target_vars[i]]
                y_pred_transformed[i][col].append(inv_tr_res_df)

        return y_pred_transformed

    @staticmethod
    def transform_from_stationary(y_pred_transformed, targets, targets_days, sma_window, nsample_forward, NTraj):
        _logger = get_logger()

        forecasts = [None] * len(targets)

        for i in range(len(targets)):
            forecasts[i] = {}

        res_type = 'pred'

        forecasts[0] = []

        for i in range(len(targets)):
            for ntraj in range(NTraj):

                frc = pd.DataFrame(data=np.zeros([y_pred_transformed[i][res_type][ntraj].shape[0],
                                                  y_pred_transformed[i][res_type][ntraj].shape[1]]),
                                   index=y_pred_transformed[i][res_type][ntraj].index,
                                   columns=[f'{targets[i]}_{j}d' for j in targets_days])

                target_sma = SMAIndicator(close=SpbDataProcessing.data[targets[i]], window=sma_window).sma_indicator()

                for idx in range(y_pred_transformed[i][res_type][ntraj].shape[0]):

                    for target_var in range(nsample_forward):
                        frc.iloc[idx, target_var] = np.exp(

                            y_pred_transformed[i][res_type][ntraj].iloc[idx, target_var] +
                            np.log(target_sma.loc[y_pred_transformed[i][res_type][ntraj].index[idx]] + 1)

                        ) - 1

                forecasts[i].append(frc.values)

                _logger.info(f'{ntraj + 1} from {NTraj} forecast done.')

        return forecasts