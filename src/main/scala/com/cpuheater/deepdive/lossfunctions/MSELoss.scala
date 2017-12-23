package com.cpuheater.deepdive.lossfunctions

import com.cpuheater.deepdive.activations.ActivationFn
import com.cpuheater.deepdive.nn.core.Activation
import com.cpuheater.deepdive.nn.core.Activation
import com.cpuheater.deepdive.nn.layers.Layer
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.dataset.api.DataSet
import shapeless.HList
import shapeless.ops.hlist.ToList
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}
import org.nd4s.Implicits._

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._



object MSELoss extends LossFunction{

  def computeLossAndGradient(label: INDArray, output: INDArray) : Float = {
    var scoreArr: INDArray = output.rsubi(label)
    scoreArr = scoreArr.muli(scoreArr)
    var score = scoreArr.sumNumber.floatValue()
    score /= scoreArr.size(0).toFloat
    score
  }

  def computeGradient(label: INDArray, output: INDArray, activationFn: ActivationFn): INDArray =  {
    var diff = (output-label) * activationFn.derivative(output)
    diff
  }

}
