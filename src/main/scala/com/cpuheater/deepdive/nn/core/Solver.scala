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

  def fit(dataSet: DataSet): Unit = {
    dataSet.batchBy(config.batchSize).zipWithIndex.foreach{
      case (batch, index) =>
        println(s"Batch ${index}")
        step(batch)
    }
  }

  def fit(iterator: DataSetIterator, alpha: Double): Unit = {
    ???
  }


  def predict(x: INDArray) = {
    model.predict(x)
  }


  private def step(dataSet: DataSet) = {
    val x = dataSet.getFeatures
    val y = dataSet.getLabels
    val (loss, grads) = model.calcGradientAndLoss(x, y)
    model.layers.zipWithIndex.foreach {
      case (layer, index) =>
        layer.params(CompType.W) = layer.params(CompType.W) - grads(s"${CompType.W}${index+1}") * config.lr
        layer.params(CompType.B) = layer.params(CompType.B) - grads(s"${CompType.B}${index+1}") * config.lr
    }
    model.layers.zipWithIndex.foreach {
      case (layer, index) =>
        println(layer.params(CompType.W))
        println(layer.params(CompType.B))
    }

  }

}
