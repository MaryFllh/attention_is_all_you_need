class Optimizer:
    # Used https://nlp.seas.harvard.edu/2018/04/03/attention.html#training-loop as ref
    def __init__(self, d_model, optimizer, warmup):
        """
        Optimizer wrapper implemening a dynamic lr with warmup
        """
        self.d_model = d_model
        self.optimizer = optimizer
        self.warmup = warmup
        self._step = 0
        self._rate = 0

    def step(self):
        """
        Updates the parameters and rate
        """
        self._step += 1
        rate = self.compute_learning_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def compute_learning_rate(self, step=None):
        """
        Computes the learning rate based on the step number
        """
        if step is None:
            step = self._step
        return (self.d_model**-0.5) * min(
            step ** (-0.5), step * self.warmup ** (-1.5)
        )
