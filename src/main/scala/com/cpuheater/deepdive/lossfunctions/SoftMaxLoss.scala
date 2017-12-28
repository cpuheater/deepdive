package com.cpuheater.deepdive.lossfunctions

import com.cpuheater.deepdive.nn.layers.Layer
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.activations.impl.ActivationSoftmax
import org.nd4j.linalg.dataset.api.DataSet
import shapeless.HList
import shapeless.ops.hlist.ToList
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT
import org.slf4j.{Logger, LoggerFactory}
import org.nd4s.Implicits._

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._



/*
shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
 */


object SoftMaxLoss  extends LossFunction2 {

  override def computeGradientAndScore(outputs: INDArray, labels: INDArray) : (Double, INDArray) = {
    val nbTrainigExamples = outputs.rows()
    val d = outputs.max(1)
    val outputsMinusMax = outputs.subColumnVector(outputs.max(1))
    val scores = exp(outputsMinusMax)
    val sum = scores.sum(1).reshape(nbTrainigExamples, 1)
    val probs = scores.divColumnVector(sum)
    val loss = -(log(probs)*labels).sumNumber().doubleValue()/nbTrainigExamples
    val grad = (probs - labels)/nbTrainigExamples
    (loss, grad)

  }

}

