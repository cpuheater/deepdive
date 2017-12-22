package com.cpuheater.deepdive.lossfunctions

import com.cpuheater.deepdive.activations.ActivationFn
import com.cpuheater.deepdive.nn.core.Activation
import com.cpuheater.deepdive.nn.core.Activation
import com.cpuheater.deepdive.nn.layers.Layer
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import shapeless.HList
import shapeless.ops.hlist.ToList
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import org.nd4s.Implicits._

object CrossEntropyLoss extends LossFunction {

  def computeScoreAndGradient(label: INDArray, output: INDArray) : Float = {
    val term1 = log(output).mul(-label)
    val term2 = log(output.rsub(1)).mul(label.rsub(1))
    Nd4j.clearNans(term2)
    term1.sub(term2).sumNumber().floatValue()
  }

  def computeGradient(label: INDArray, output: INDArray, activationFn: ActivationFn): INDArray =  {
    val diff = output-label
    diff
  }


}
