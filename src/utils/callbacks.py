class EarlyStoppingCallback:
    """
    Custom early stopping implementation.
    One the first __call__ the last_loss and last_f1 values are set.
    Each subsequent call will check if it should increment the trigger_times value.
    When maximum patience is reached, the early_stop value is set to True as this value
    should be checked to stop the training process.
    If a current_loss (and current_f1) value is lower than the stored last_loss (and last_f1),
    the trigger will be reset.
    """
    def __init__(self, patience: int = 5, incl_loss: bool = False):
        self.patience: int = patience
        self.incl_loss: bool = incl_loss
        self.start: bool = True
        self.trigger_times: int = 0
        self.last_loss: float = 0.0
        self.last_f1: float = 0.0
        self.best_model: str = ''
        self.early_stop: bool = False

    def __call__(self, current_loss: float, current_f1: float, model_dir: str = None) -> None:
        if self.start:
            self.set_properties(current_loss, current_f1, model_dir)
            self.start = False
        else:
            if self.incl_loss and current_loss > self.last_loss and self.last_f1 > current_f1:
                self.incr_trigger()
                self.check_early_stop()

            elif current_f1 < self.last_f1:
                self.incr_trigger()
                self.check_early_stop()

            else:
                self.reset_trigger()
                self.set_properties(current_loss, current_f1, model_dir)

    def incr_trigger(self) -> None:
        self.trigger_times += 1

    def reset_trigger(self) -> None:
        self.trigger_times = 0

    def set_properties(self, current_loss: float, current_f1: float, model_dir: str) -> None:
        self.last_loss = current_loss
        self.last_f1 = current_f1
        self.best_model = model_dir

    def check_early_stop(self) -> None:
        if self.trigger_times >= self.patience:
            self.early_stop = True

    def epoch_summary(self) -> str:
        return f'\n****** Early Stopping ******\n'\
               f'Counter = {self.trigger_times}\n'\
               f'Patience = {self.patience}\n'\
               f'Validation micro-F1 (BEST) = {self.last_f1}\n'\
               f'Model save path (BEST) = {self.best_model}\n'\
               f'****************************\n'

    def stop(self) -> bool:
        return self.early_stop

    def reset(self) -> None:
        self.start: bool = True
        self.trigger_times: int = 0
        self.last_loss: float = 0.0
        self.last_f1: float = 0.0
        self.best_model: str = ''
        self.early_stop: bool = False


