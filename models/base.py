import abc
# import logging

# LOGGER = logging.Logger("IncLearn", level="INFO")


class GAN(abc.ABC):

    def __init__(self, *args, **kwargs):
        pass


    def train(self, train_loader, test_loader, output_dir, train_output_dir, test_output_dir, model_output_dir):
#         LOGGER.info("train task")
        self._train(train_loader, test_loader, output_dir, train_output_dir, test_output_dir, model_output_dir)

    def _train(self, train_loader, test_loader, output_dir, train_output_dir, test_output_dir, model_output_dir):
        raise NotImplementedError

