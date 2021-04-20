from tensorboardX import SummaryWriter

class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardWriter, self).__init__(logdir)

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_evaluation(self, devel_loss, devel_wer, step):
        self.add_scalar('devel_loss', devel_loss, step)
        self.add_scalar('devel_wer', devel_wer, step)