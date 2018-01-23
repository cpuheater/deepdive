package com.cpuheater.deepdive.nn.core

import com.cpuheater.deepdive.lossfunctions.{LossFunction, LossFunction2, SoftMaxLoss}
import com.cpuheater.deepdive.nn.layers.{GradResult, Layer, LinearLayer, ParamType}

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

class SequentialModel(val layers: List[Layer]) {


  private val hiddenLayers = layers.reverse.tail.reverse
  private val outputLayer = layers.last


  private def forward(x: INDArray): List[INDArray] = {
    val scores = hiddenLayers.foldLeft(List(x)){
      case (accum, layer) =>
        val newScore = layer.forward(accum.head)
        newScore::accum
    }

    outputLayer.forward(scores.head)::scores
  }

  def forwardAndBackwardPass(x: INDArray, y: INDArray): (Double, Map[String, INDArray]) = {

    val scores = forward(x)

    val (loss, dout) = SoftMaxLoss.computeGradientAndScore(scores.head, y)

    val GradResult(dx, g) = outputLayer.backward(scores.tail.head, dout)

    val grads = scala.collection.mutable.Map[String, INDArray](g.toSeq: _*)

    hiddenLayers.reverse.zip(scores.tail.tail).foldLeft(dx){
      case (dprev, (layer, score)) =>
        val GradResult(dx, g) = layer.backward(score, dprev)
        grads.putAll(g)
        dx
    }
    (loss, grads.toMap)
  }


  def predict(x: INDArray): INDArray =  {
    forward(x).head
  }


}
