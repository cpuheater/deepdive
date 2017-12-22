package com.cpuheater.deepdive.nn.core

import com.cpuheater.deepdive.nn.layers.Layer
import com.cpuheater.deepdive.lossfunctions.LossFunction
import com.cpuheater.deepdive.nn.layers.Layer
import com.cpuheater.deepdive.lossfunctions.LossFunction
import com.cpuheater.deepdive.nn.LayerConfig
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


class FeedForwardNetwork(layers: List[LayerConfig], lossFn: LossFunction)  {


  private val log: Logger = LoggerFactory.getLogger(this.getClass)

  val numLayers = layers.length


  private var biases = layers.foldLeft(Seq.empty[INDArray]){
    case (accum, layer) =>
      accum :+ Nd4j.ones(layer.nbOutput, 1)

  }

  private var weights = layers.foldLeft(Seq.empty[INDArray]){
    case (accum, layer)  =>
      val epsilonInit = 0.12
      accum :+ Nd4j.rand(layer.nbOutput, layer.nbInput)  * 2 * epsilonInit - epsilonInit
  }

  private def feedForward(feature: INDArray): (Array[INDArray], Array[INDArray]) = {
    biases.zip(weights).foldLeft((Array.empty[INDArray], Array(feature))) {
      case ((zs, activations), (bias, weight)) =>
        val z = weight.dot(activations.last) + bias
        val newActivation = sigmoid(z)
        (zs :+ z , activations :+ newActivation)
    }
  }

  private def backprop(feature: INDArray, label: INDArray) = {
    val biasesGrad = new Array[INDArray](numLayers)
    val weightsGrad = new Array[INDArray](numLayers)

    val (_, activations) =  feedForward(feature)


    val deltaTMP = (activations.last - label) *  layers.last.activation.derivative(activations.last)
    val delta = lossFn.computeGradient(label, activations.last, layers.last.activation)

    biasesGrad.update(numLayers-1, delta)
    weightsGrad.update(numLayers-1, delta.dot(activations.reverse.tail.head.T))

    (numLayers-1 until  0 by -1).foldLeft(delta){
      case (delta, l) =>
        val newDelta = weights(l).T.dot(delta) * layers(l).activation.derivative(activations(l))
        weightsGrad(l-1) =  newDelta.dot(activations(l-1).T)
        biasesGrad(l-1) = newDelta
        newDelta
    }

    (weightsGrad, biasesGrad)

  }



  private  def optimize(features: INDArray, labels: INDArray, alpha: Double) = {
    var totalBiasesGrad = biases.map( bias => Nd4j.zeros(bias.shape() : _*))
    var totalWeightsGrad = weights.map(weight => Nd4j.zeros(weight.shape(): _*))
    val miniBatchSize = features.rows()

    (0 until labels.rows()).map{
      index =>
        val (weightsGrad, biasesGrad) =  backprop(features.getRow(index).T,  labels.getRow(index).T)
        totalWeightsGrad = totalWeightsGrad.zip(weightsGrad).map{
          case (total, current) =>
            total + current
        }
        totalBiasesGrad = totalBiasesGrad.zip(biasesGrad).map{
          case (total, current) =>
            total + current
        }
    }
    weights = weights.zip(totalWeightsGrad).map{
      case (weight, weightGrad) =>
        weight - weightGrad * (alpha/miniBatchSize)
    }
    biases = biases.zip(totalBiasesGrad).map{
      case (bias, biasGrad) =>
        bias -  biasGrad * (alpha/miniBatchSize)
    }
  }


  def predict(x: INDArray):  INDArray = {
    val (_, activations) =  feedForward(x)
    activations.last
  }



  def fit(dataSet: DataSet, epochs: Int, batchSize: Int, alpha: Double): Unit = {
    (0 until epochs).foreach{
      i =>
        println(s"Epoch ${i}")
        dataSet.shuffle()
        val batches = dataSet.batchBy(batchSize)
        batches.foreach{
          batch: DataSet =>
            val labels = batch.getLabels
            val features = batch.getFeatures
            optimize(features, labels, alpha)
        }
    }
  }


  def fit(iterator: DataSetIterator, alpha: Double): Unit = {
    var continue = iterator.hasNext
    while(continue){
      val next = iterator.next()
      //next.shuffle()
      if(next.getFeatures == null ||  next.getLabels == null) {
        continue = false
      } else {
         optimize(next.getFeatures, next.getLabels, alpha)
         continue = iterator.hasNext
      }

    }
    if(iterator.resetSupported())
      iterator.reset()
  }


}
