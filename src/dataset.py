import pandas as pd
from os.path import join
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class DataSet:
    def __init__(self, datapath='../data', frac=1, adddim=False, center=False, pca_comps=None, poly=None, train_on_valid=False,
                 train_on_noise=None, balance=True):

        self.datapath = datapath
        self.frac = frac
        self.adddim = adddim
        self.center = center
        self.pca_comps = pca_comps
        self.poly = poly
        self.train_on_valid = train_on_valid
        self.balance = balance
        # INS data
        training_data = pd.read_csv(join(datapath, 'train-1.csv'), header=0)
        if frac:
            training_data = training_data.sample(frac=frac)  # randomly rearrange rows when 1
        if balance:
            training_data = self._balance(training_data)

        self.training_data = training_data

        # OOS data
        prediction_data = pd.read_csv(join(datapath, 'test-1.csv'), header=0)
        self.prediction_data = prediction_data

        if train_on_valid:
            print('good luck cheating! (training_on_validation)')
            self.training_data = pd.concat([self.training_data, self.prediction_data.dropna()]).reset_index(drop=True)


        features = [f for f in list(self.training_data.columns) if "feature" in f]


        if train_on_noise:
            extra_noise = np.random.randn(*self.training_data[features].shape)

            try:
                extra_noise *= train_on_noise
            except:
                extra_noise *= .01

            training_with_noise = self.training_data.copy()
            training_with_noise[features] = self.training_data[features] + extra_noise

            self.training_data = self.training_data.append(training_with_noise).reset_index(drop=True)


        X_train = self.training_data[features]

        # dimensionality reduction
        if self.pca_comps:
            from sklearn.decomposition import PCA
            self.pca = PCA(whiten=False)
            self.pca.fit(X_train)
            #print('pca explained var ratio:', self.pca.explained_variance_ratio_)

        if poly:
            self.poly = PolynomialFeatures(poly, interaction_only=True, include_bias=False)
            print('adding', poly, 'th order features')


        self.X_train = self._preprocess(X_train)
        self.Y_train = self.training_data["target"].values


        x_valid = prediction_data[prediction_data['target'] >= 0][features]
        self.x_valid = self._preprocess(x_valid)
        self.y_valid = self.prediction_data["target"].dropna().values

        x_prediction = self.prediction_data[features]
        self.x_prediction = self._preprocess(x_prediction)
        self.ids = self.prediction_data["id"]
        self.ids_train = self.training_data["id"]

        if pca_comps or poly:
            # update training data to reflect this
            training_data_features = self.X_train.reshape(len(self.training_data), self.X_train.shape[1])
            training_data_features = pd.DataFrame(training_data_features,
                                                  columns=['feature_prepro_' + str(x) for x in range(self.X_train.shape[1])])
            self.training_data = pd.concat([self.training_data[['id', 'era', 'target']],
                                            training_data_features], axis=1)

            # update prediction data to reflect this
            prediction_data_features = self.x_prediction.reshape(len(self.prediction_data), self.x_prediction.shape[1])
            prediction_data_features = pd.DataFrame(prediction_data_features,
                                                    columns=['feature_prepro_' + str(x) for x in range(self.x_prediction.shape[1])])

            self.prediction_data = pd.concat([self.prediction_data[['id', 'era', 'target', 'data_type']],
                                              prediction_data_features], axis=1)


        self.num_features = self.X_train.shape[1]
        assert(self.num_features==self.x_prediction.shape[1])
        self._print_shapes()


    def _print_shapes(self):
        print('Shapes:')
        print('X_train:', self.X_train.shape)
        print('training_data', self.training_data.shape)
        print(self.training_data.columns)

        print('x_prediction:', self.x_prediction.shape)

    def _preprocess(self, X):
        if self.center:
            X-= X.mean()

        X = X.values

        if self.pca_comps:
            X = self.pca.transform(X)[:, self.pca_comps]

        if self.poly:
            X = self.poly.fit_transform(X)

        if self.adddim:
            X = X[:, :, np.newaxis] # could also do None

        return X

    def _balance(self, training_era):
        training_era1 = training_era[training_era.target==1]
        training_era0 = training_era[training_era.target==0]
        zero_to_one_ratio = float(len(training_era0))/len(training_era1)
        if zero_to_one_ratio<1:
            training_era1 = training_era1.sample(frac=zero_to_one_ratio)
        else:
            training_era0 = training_era0.sample(frac=1./zero_to_one_ratio)
        training_era = pd.concat([training_era0, training_era1]).sample(frac=1)
        return training_era

    def training_by_era(self, eras, balance=True):
        training_era = self.training_data[self.training_data['era'].isin(eras)]
        if balance:
            training_era = self._balance(training_era).dropna()
        features = [f for f in list(self.training_data.columns) if "feature" in f]
        X_era = training_era[features].values

        if self.adddim:
            X_era = X_era[:, :, np.newaxis] # could also do None
        #X_era = self._preprocess(X_era)
        Y_era = training_era["target"].values
        return X_era, Y_era


    def get_train_and_val(self):
        return self.X_train, self.Y_train, self.x_valid, self.y_valid