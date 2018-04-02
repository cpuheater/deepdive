package com.cpuheater.deepdive.nn.core

import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import com.cpuheater.deepdive.nn.layers.ParamType
import com.cpuheater.deepdive.optimize.BaseOptimizer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.BaseOp
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.collection.mutable

class Solver(model: SequentialModel,
             config: BuildConfig) extends SolverSupport {

  private val optimizer = buildOptimizer(config, model.layers.flatMap(_.params).toMap)

  def fit(dataSet: DataSet, batchSize: Int, epochs:Int): Unit = doEpoch(epochs){
    dataSet.batchBy(batchSize).zipWithIndex.foreach{
      case (batch, index) =>
        println(s"Batch ${index}")
        val x = batch.getFeatures
        val y = batch.getLabels
        step(x, y)
    }
  }

  def fit(iterator: DataSetIterator, batchSize: Int, epochs:Int): Unit = doEpoch(epochs){
      var continue = iterator.hasNext
      while(continue){
        val next = iterator.next()
        if(next.getFeatures == null ||  next.getLabels == null) {
          continue = false
        } else {
          step(next.getFeatures, next.getLabels)
          continue = iterator.hasNext
        }

      }
      if(iterator.resetSupported())
        iterator.reset()
  }

  private def doEpoch[T](epochs: Int)(f : => Unit) = {
    for (i <- 1 to epochs) {
      println(s"Epoch ${i}")
      f
    }
  }

  def evaluate(x: INDArray, y: INDArray) = {

  }

  def predict(x: INDArray) = {
    model.predict(x)
  }

  private def step(x: INDArray, y: INDArray): Unit = {
    val (loss, grads) = model.forwardAndBackwardPass(x, y)
    println(s"loss: $loss")
    model.layers.zipWithIndex.foreach {
      case (layer, index) =>
        val wKey = ParamType.toString(ParamType.W, index+1)
        val bKey = ParamType.toString(ParamType.B, index+1)
        layer.params(wKey) -= optimizer.optimize(layer.params(wKey), grads(wKey), wKey)
        layer.params(bKey) -= optimizer.optimize(layer.params(bKey), grads(bKey), bKey)
    }

  }

  def params(): Map[String, INDArray] = {
    model.layers.flatMap(_.params).toMap
  }

}
