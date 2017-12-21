package com.cpuheater.deepdive.nn.core

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

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._

class Solver(model: MultiLayerNetwork) {


  var epoch = 0
  var best_val_acc = 0
  var best_params = Map.empty[String, INDArray]
  var loss_history = List.empty[Double]
  var train_acc_history = List.empty[Double]
  var val_acc_history = List.empty[Double]


  def step(dataSet: DataSet) = {
    val x = dataSet.getFeatures
    val y = dataSet.getLabels
    val (loss, grads) = model.loss(x, y)
    println(grads)

  }

  def train(dataSet: DataSet, lrDecay: Double = 1.0, batchSize: Int = 2) = {

    dataSet.batchBy(batchSize).zipWithIndex.map{
      case (batch, index) =>
        println(s"Batch ${index}")
        step(batch)
    }

  }




}
