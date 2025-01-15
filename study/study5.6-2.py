import torch
from torch import nn
from d2l import torch as d2l
from triton.language import tensor


class ActivationsVariance(d2l.Classifier):
    def __init__(self, trainer, layer):
        super().__init__()
        self.trainer = trainer
        self.board = d2l.ProgressBoard()
        self.layer = layer

    def forward(self, X):
        if not self.training:
            return X

        variance = torch.var(X, dim=1, unbiased=False, keepdim=False)
        meanVar = torch.mean(variance)

        print(meanVar)

        self.board.xlabel = 'epoch'

        x = self.trainer.train_batch_idx / \
            self.trainer.num_train_batches
        n = self.trainer.num_train_batches / \
            self.plot_train_per_epoch

        self.board.draw(x, d2l.numpy(d2l.to(meanVar, d2l.cpu())),
                        'activations_variance_' + str(self.layer),
                        every_n=int(n))
        return X


trainer = d2l.Trainer(max_epochs=10)


class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.net = nn.Sequential(
            nn.Flatten(), self.lin1, nn.ReLU(), ActivationsVariance(trainer, 1),
            self.lin2, nn.ReLU(),
            self.lin3)


hparams = {'num_outputs': 10, 'num_hiddens_1': 256, 'num_hiddens_2': 256,
           'dropout_1': 0.5, 'dropout_2': 0.5, 'lr': 0.1}
model = MLP(**hparams)
data = d2l.FashionMNIST(batch_size=256)
data.num_workers = 0
trainer.fit(model, data)

d2l.plt.show()

print('===========================')


class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.net = nn.Sequential(
            nn.Flatten(), self.lin1, nn.ReLU(), ActivationsVariance(trainer, 1),
            nn.Dropout(dropout_1), self.lin2, nn.ReLU(),
            nn.Dropout(dropout_2), self.lin3)

    def loss(self, Y_hat, Y, averaged=True):
        l = super().loss(Y_hat, Y, averaged=averaged)
        return l

    # def configure_optimizers(self):
    #     bias_params = [p for name, p in self.named_parameters() if 'bias' in name]
    #     others = [p for name, p in self.named_parameters() if 'bias' not in name]
    #
    #     return torch.optim.SGD([
    #         {'params': others},
    #         {'params': bias_params, 'weight_decay': 0},
    #     ], lr=self.lr, weight_decay=0.0001)


model = DropoutMLP(**hparams)
trainer.fit(model, data)

d2l.plt.show()
