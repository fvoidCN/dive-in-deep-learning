import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


def download(url, folder, sha1_hash=None):
    """Download a file to folder and return the local filepath."""


def extract(filename, folder):
    """Extract a zip/tar file into folder."""


class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))


data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)

print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])


class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.net = nn.Sequential(
            nn.Flatten(), self.lin1, nn.ReLU(),
            nn.Dropout(dropout_1), self.lin2, nn.ReLU(),
            nn.Dropout(dropout_2), self.lin3)

    def loss(self, y_hat, y, averaged=True):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    # def configure_optimizers(self):
    #     bias_params = [p for name, p in self.named_parameters() if 'bias' in name]
    #     others = [p for name, p in self.named_parameters() if 'bias' not in name]
    #
    #     return torch.optim.SGD([
    #         {'params': others},
    #         {'params': bias_params, 'weight_decay': 0},
    #     ], lr=self.lr, weight_decay=0.001)


@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Standardize numerical columns
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding
    features = pd.get_dummies(features, dummy_na=True)
    # Save preprocessed features
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()


data.preprocess()
print(data.train.shape)

hparams = {'num_outputs': 1, 'num_hiddens_1': 512, 'num_hiddens_2': 256,
           'dropout_1': 0, 'dropout_2': 0, 'lr': 0.015}


@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: torch.tensor(x.values.astype(float),
                                        dtype=torch.float32)
    # Logarithm of prices
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               torch.log(get_tensor(data[label])).reshape((-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)


def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j + 1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),
                                data.train.loc[idx]))
    return rets


def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = DropoutMLP(**hparams)
        model.board.yscale = 'log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss) / len(val_loss)}')
    return models


trainer = d2l.Trainer(max_epochs=20)
models = k_fold(trainer, data, k=5, lr=0.01)

d2l.plt.show()

preds = [model(torch.tensor(data.val.values.astype(float), dtype=torch.float32))
         for model in models]
# Taking exponentiation of predictions in the logarithm scale
ensemble_preds = torch.exp(torch.cat(preds, 1)).mean(1)
submission = pd.DataFrame({'Id': data.raw_val.Id,
                           'SalePrice': ensemble_preds.detach().numpy()})
submission.to_csv('submission.csv', index=False)
