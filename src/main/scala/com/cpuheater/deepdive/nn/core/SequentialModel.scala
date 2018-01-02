package com.cpuheater.deepdive.nn.core

import com.cpuheater.deepdive.lossfunctions.{LossFunction, LossFunction2, SoftMaxLoss}
import com.cpuheater.deepdive.nn.layers.{ParamType, Layer, LinearLayer}
import com.cpuheater.deepdive.nn.core.FeedForwardNetwork
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

class SequentialModel(val layers: List[Layer]) {


  private val hiddenLayers = layers.reverse.tail.reverse
  private val outputLayer = layers.last


  private def forward(x: INDArray): INDArray = {
    val lastHiddenScore = hiddenLayers.foldLeft(x){
      case (score, layer) =>
        val newScore = layer.forward(score)
        newScore
    }

    outputLayer.forward(lastHiddenScore)
  }

  def forwardAndBackwardPass(x: INDArray, y: INDArray): (Double, Map[String, INDArray]) = {

    val preOutput = forward(x)

    val (loss, dout) = SoftMaxLoss.computeGradientAndScore(preOutput, y)

    val (dx, dw, db) = outputLayer.backward(dout)

    val grads = scala.collection.mutable.Map[String, INDArray]()
    grads(s"${ParamType.W}${layers.length}") = dw
    grads(s"${ParamType.B}${layers.length}") = db

    hiddenLayers.reverse.zip(hiddenLayers.length to 1 by -1).foldLeft(dx){
      case (dprev, (layer, index)) =>
        val (dx, dw, db) = layer.backward(dprev)
        grads(s"${ParamType.W}${index}") = dw
        grads(s"${ParamType.B}${index}") = db
        dx
    }
    (loss, grads.toMap)
  }


  def predict(x: INDArray): INDArray =  {
    forward(x)
  }


}
