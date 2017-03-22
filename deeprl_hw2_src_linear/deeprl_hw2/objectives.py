"""Loss functions."""

import tensorflow as tf
import semver


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    absDiff = tf.abs(y_true - y_pred)
    condition = tf.less(absDiff,max_grad)
    smallDiff = .5*tf.square(absDiff)
    largeDiff = max_grad*absDiff-.5*pow(max_grad,2)
    return tf.select(condition,smallDiff,largeDiff)


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """

    absDiff = tf.abs(y_true - y_pred)
    condition = tf.less(absDiff,max_grad)
    smallDiff = .5*tf.square(absDiff)
    largeDiff = max_grad*absDiff-.5*pow(max_grad,2)
    return tf.reduce_mean(tf.where(condition,smallDiff,largeDiff))
    
