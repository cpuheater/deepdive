package com.cpuheater.deepdive.lossfunctions

import com.cpuheater.deepdive.activations.{ActivationFn, Identity}
import com.cpuheater.deepdive.nn.layers.Layer
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.activations.impl.ActivationSoftmax
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax
import org.nd4j.linalg.dataset.api.DataSet
import shapeless.HList
import shapeless.ops.hlist.ToList
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT
import org.slf4j.{Logger, LoggerFactory}
import org.nd4s.Implicits._

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._


object SoftMaxLoss  extends LossFunction2 {


  private val softmaxClipEps = 1.0E-10

  override def computeGradientAndScore(outputs: INDArray,
                                       labels: INDArray, activationFn: ActivationFn = Identity) : (Double, INDArray) = {

    /*
    if (labels.size(1) != preOutput.size(1)) {
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1)
                            + ") does not match output layer" + " number of outputs (nOut = " + preOutput.size(1)
                            + ") ");

        }
     */

    val nbTrainigExamples = outputs.rows()
    val outputsMinusMax = outputs.subColumnVector(outputs.max(1))
    val scores = exp(outputsMinusMax)
    val sum = scores.sum(1).reshape(nbTrainigExamples, 1)
    val probs = scores.divColumnVector(sum)

    BooleanIndexing.replaceWhere(probs, softmaxClipEps, Conditions.lessThan(softmaxClipEps))
    BooleanIndexing.replaceWhere(probs, 1.0 - softmaxClipEps, Conditions.greaterThan(1.0 - softmaxClipEps))

    val loss = -(log(probs)*labels).sumNumber().doubleValue()/nbTrainigExamples
    val grad = (probs - labels)/nbTrainigExamples
    (loss, grad)

  }

}

