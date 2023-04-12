from tensorboardX import SummaryWriter # have to install pip3 install tensorboardX
import torch
import os
import errno

class Logger:
    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = f'{model_name}_{data_name}'
        self.data_subdir = f'{model_name}/{data_name}'

        self.writer = SummaryWriter(comment=self.comment)


    def loss_log(self, train_loss, val_loss, nth_epoch):
        self.writer.add_scalars(
            'Train vs Val Loss',
            {'Training': train_loss, 'Validation': val_loss},
            nth_epoch
        )
        
    def acc_log(self, train_acc, val_acc, nth_epoch):
        self.writer.add_scalars(
            'Train vs Val Accuracy',
            {'Training': train_acc, 'Validation': val_acc},
            nth_epoch
        )

    def save_models(self, model, nth_epoch):
        out_dir = f'saved/{self.model_name}'
        Logger._make_dir(out_dir)
        torch.save(
            model.state_dict(),
            f'{out_dir}/Ep.{nth_epoch}.pth'
        )

    def close(self):
        self.writer.close()

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @staticmethod
    def _step(epoch):
        pass
