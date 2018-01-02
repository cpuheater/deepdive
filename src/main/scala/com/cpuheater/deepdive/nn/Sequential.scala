package com.cpuheater.deepdive.nn


import com.cpuheater.deepdive.lossfunctions.LossFunction2
import com.cpuheater.deepdive.nn.layers.{CompType, ConvLayer, LinearLayer}
import com.cpuheater.deepdive.nn.core.{Config, SequentialModel, Solver}
import com.cpuheater.deepdive.weights.WeightsInitializer
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable

class Sequential protected (layers: List[LayerConfig] = Nil) {

   def add(layer: LayerConfig)  = new Sequential(layers:+layer)

   def build(loss: LossFunction2,
             lr: Double,
             batchSize: Int,
             seed: Option[Int] = None, numOfEpoch:Int = 2) : Solver = {

     seed.foreach(Nd4j.getRandom.setSeed)

     val newLayers = layers.zip(1 to layers.length).map{
       case (linearConfig: Linear, index) =>
         val w = WeightsInitializer.initWeights(
           WeightsInitType.UNIFORM,
           Array(linearConfig.nbInput,
           linearConfig.nbOutput))
         val b = Nd4j.zeros(linearConfig.nbOutput)
         val params = mutable.Map[String, INDArray](CompType.print(CompType.W, index) -> w,
           CompType.print(CompType.B, index) -> b)
         new LinearLayer(linearConfig, params, index)

       case (convConfig: Conv, index) =>

         val w = WeightsInitializer.initWeights(
           WeightsInitType.UNIFORM,
           Array(convConfig.nbOfFilters,
             convConfig.channels,
             convConfig.filterHeight,
             convConfig.filterWidth))
         val b = Nd4j.zeros(convConfig.nbOfFilters)
         val params = mutable.Map[String, INDArray](CompType.print(CompType.W, index) -> w,
           CompType.print(CompType.B, index) -> b)

         new ConvLayer(convConfig, params, index)

     }


     val model = new SequentialModel(newLayers)
     val config = Config(layers, loss, lr, batchSize, numOfEpoch = numOfEpoch)
     new Solver(model, config)
   }

}



object Sequential {

  def apply() : Sequential = {
    new Sequential()
  }

}

