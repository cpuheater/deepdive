package com.cpuheater.deepdive.nn.core

import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import com.cpuheater.deepdive.nn.layers.CompType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._

class Solver(model: SequentialModel, config: Config) {

  def fit(dataSet: DataSet): Unit = doEpoch{
    dataSet.batchBy(config.batchSize).zipWithIndex.foreach{
      case (batch, index) =>
        println(s"Batch ${index}")
        val x = batch.getFeatures
        val y = batch.getLabels
        step(x, y)
    }
  }


  def fit(iterator: DataSetIterator): Unit = doEpoch{
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


  private def doEpoch[T](f : => Unit) =
    for(i <- 1 to config.numOfEpoch){
       println(s"Epoch ${i}")
       f
    }



  def predict(x: INDArray) = {
    model.predict(x)
  }


  private def step(x: INDArray, y: INDArray):Unit = {
    val (loss, grads) = model.forwardAndBackwardPass(x, y)
    model.layers.zipWithIndex.foreach {
      case (layer, index) =>
        val wKey = CompType.print(CompType.W, index+1)
        val bKey = CompType.print(CompType.B, index+1)
        layer.params(wKey) = layer.params(wKey) - grads(wKey) * config.lr
        layer.params(bKey) = layer.params(bKey) - grads(bKey) * config.lr
    }

  }


  def params(): Map[String, INDArray] = {
    model.layers.flatMap(_.params).toMap
  }

}
