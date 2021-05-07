"""
Logging
"""
import logging
import shutil

import torch


class Logging:
    """
    Logging
    """

    def __init__(self, path, channel):
        """
        This function is used to initialize additional attributes.
        For example, it is used to init the paths for the local logging.
        """
        self.path = path
        if path:
            self.channel = channel
            # pylint: disable=line-too-long
            self.checkpoint_path = self.path + f"/current_checkpoint_p{self.channel}.pt"  # noqa: E501
            self.best_model_path = self.path + f"/best_model_p{self.channel}.pt"  # noqa: E501
            self.loss_file_path = self.path + f"/loss_p{self.channel}.txt"
            self.log_file = self.path + f"/p{self.channel}.log"
            print("log file: ", self.log_file)
            # logger = logging.getLogger()
            # logger.setLevel(logging.INFO)
            # pylint: disable=line-too-long
            # file_handler = logging.FileHandler(self.log_file)
            # file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))  # noqa: E501
            # logger.addHandler(file_handler)
            self._clear_log()

    def _clear_log(self):
        """
        Method to clear the logs.
        """
        if self.path:
            open(self.loss_file_path, 'w').close()
            open(self.log_file, 'w').close()
            self.log(f'Training channel_p{self.channel}')
            self.save_loss_to_file('training_loss', 'validation_loss')

    @staticmethod
    def load_ckp(checkpoint_fpath: str, model, optimizer):
        """
        :param checkpoint_fpath: path to save checkpoint
        :param model: model that we want to load checkpoint parameters into
        :param optimizer: optimizer we defined in previous training
        :return
        """
        # load check point
        checkpoint = torch.load(checkpoint_fpath)
        # initialize state_dict from checkpoint to model
        model.load_state_dict(checkpoint['state_dict'])
        # initialize optimizer from checkpoint to optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        # initialize valid_loss_min from checkpoint to valid_loss_min
        # return model, optimizer
        return model, optimizer

    def save_loss_to_file(self, train_loss: any, val_loss: any):
        """
        Saves training and validation loss of one epoch to file
        :param train_loss:
        :param val_loss:
        """
        # Code for printing to a file
        if self.path:
            loss_file = open(self.loss_file_path, 'a')
            print(str(train_loss) + ',' + str(val_loss), file=loss_file)
            loss_file.close()

    @staticmethod
    def save_to_file(file_path: str, content: str):
        """
        Save the content in the specified file
        :param file_path: str
        :param content: str
        """
        # Code for printing to a file
        if file_path:
            file_name = open(file_path, 'a')
            print(str(content), file=file_name)
            file_name.close()

    @staticmethod
    def log(content: str, level='info'):
        """

        :param level:
        :param content:
        :return:
        """
        # with customized method:
        # file_path = file_path if file_path else self.log_file
        # self.save_to_file(file_path, content)
        levels = {
            'debug': logging.debug,
            'info': logging.info,
            'warning': logging.warning,
            'error': logging.error,
            'critical': logging.critical,
        }
        levels[level](content)

    def save_ckp(self, state, is_best: bool):
        """

        :param state: checkpoint we want to save
        :param is_best: is this the best checkpoint; min validation loss
        """
        # save checkpoint data to the path given, checkpoint_path
        if self.path:
            torch.save(state, self.checkpoint_path)
            # if it is a best model, min validation loss
            if is_best:
                # copy that checkpoint file to best path given, best_model_path
                shutil.copyfile(self.checkpoint_path, self.best_model_path)
