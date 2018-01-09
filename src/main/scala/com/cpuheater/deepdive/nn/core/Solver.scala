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


  def fit(dataSet: DataSet): Unit = doEpoch{
    optimizer =>

    dataSet.batchBy(config.batchSize).zipWithIndex.foreach{
      case (batch, index) =>
        println(s"Batch ${index}")
        val x = batch.getFeatures
        val y = batch.getLabels
        step(x, y, optimizer)
    }
  }


  def fit(iterator: DataSetIterator): Unit = doEpoch{
    optimizer =>
      var continue = iterator.hasNext
      while(continue){
        val next = iterator.next()
        if(next.getFeatures == null ||  next.getLabels == null) {
          continue = false
        } else {
          step(next.getFeatures, next.getLabels, optimizer)
          continue = iterator.hasNext
        }

      }
      if(iterator.resetSupported())
        iterator.reset()
  }

  private def doEpoch[T](f : BaseOptimizer=> Unit) = {
    val optimizer = buildOptimizer(config, model.layers.flatMap(_.params).toMap)
    for (i <- 1 to config.numOfEpoch) {
      println(s"Epoch ${i}")
      f(optimizer)
    }
  }


  def predict(x: INDArray) = {
    model.predict(x)
  }

  private def step(x: INDArray, y: INDArray, optimizer: BaseOptimizer): Unit = {
    val (loss, grads) = model.forwardAndBackwardPass(x, y)
    println(s"loss: $loss")
    model.layers.zipWithIndex.foreach {
      case (layer, index) =>
        val wKey = ParamType.toString(ParamType.W, index+1)
        val bKey = ParamType.toString(ParamType.B, index+1)
        layer.params(wKey) = optimizer.optimize(layer.params(wKey), grads(wKey), wKey)
        layer.params(bKey) = optimizer.optimize(layer.params(bKey), grads(bKey), bKey)
    }

  }


  def params(): Map[String, INDArray] = {
    model.layers.flatMap(_.params).toMap
  }

}
