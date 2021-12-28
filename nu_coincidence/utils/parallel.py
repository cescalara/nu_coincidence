from abc import abstractmethod, ABCMeta
from joblib._parallel_backends import LokyBackend


class MultiCallback:
    """
    Allow for multiple async callbacks
    in your custom parallel backend.
    """

    def __init__(self, *callbacks):

        self.callbacks = [cb for cb in callbacks if cb]

    def __call__(self, out):

        for cb in self.callbacks:

            cb(out)


class ImmediateResultBackend(LokyBackend, metaclass=ABCMeta):
    """
    Custom backend for acting on results
    as they are processed in a joblib
    Parallel() call.
    """

    def callback(self, future):
        """
        The extra callback passes a future to
        future_handler, which must be implemented.
        """

        self.future_handler(future)

        del future

    def apply_async(self, func, callback=None):
        """
        Override this method to Handle your new
        callback in addition to any existing ones.
        """

        callbacks = MultiCallback(callback, self.callback)

        return super().apply_async(func, callbacks)

    @abstractmethod
    def future_handler(self, future):
        """
        Do something useful with the
        completed future e.g. write to file.
        """

        raise NotImplementedError()


class FileWritingBackend(ImmediateResultBackend):
    """
    Assumes result from future has a write()
    method which is called.
    """

    def future_handler(self, future):

        result = future.result()[0]

        result.write()

        del result
