package com.cpuheater.deepdive.nn


import com.cpuheater.deepdive.lossfunctions.LossFunction2
import com.cpuheater.deepdive.nn.layers._
import com.cpuheater.deepdive.nn.core._
import com.cpuheater.deepdive.nn.serialize.ModelSerializer
import com.cpuheater.deepdive.weights.WeightsInitializer

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable

class Sequential protected (layers: List[LayerConfig] = Nil)  {

   def add(layer: LayerConfig)  = new Sequential(layers:+layer)

   def compile(loss: LossFunction2,
               optimizerConfig: Optimizer = Optimizer.SGD(),
               seed: Option[Int] = None) : Solver = {

     seed.foreach(Nd4j.getRandom.setSeed)

     val newLayers = layers.zip(1 to layers.length).map{
       case (linearConfig: Linear, index) =>
         val w = WeightsInitializer.initWeights(
           WeightsInitType.NORMAL,
           Array(linearConfig.nbInput,
           linearConfig.nbOutput))
         val b = Nd4j.zeros(linearConfig.nbOutput)
         val params = mutable.Map[String, INDArray](ParamType.toString(ParamType.W, index) -> w,
           ParamType.toString(ParamType.B, index) -> b)
         new LinearLayer(linearConfig, params, index)

       case (convConfig: Conv2d, index) =>

         val w = WeightsInitializer.initWeights(
           WeightsInitType.NORMAL,
           Array(convConfig.numFilters,
             convConfig.channels,
             convConfig.filterHeight,
             convConfig.filterWidth))
         val b = Nd4j.zeros(convConfig.numFilters)
         val params = mutable.Map[String, INDArray](ParamType.toString(ParamType.W, index) -> w,
           ParamType.toString(ParamType.B, index) -> b)

         new ConvLayer(convConfig, params, index)

       case (maxPoolConfig: MaxPool, index) =>
         
         new MaxPoolLayer(maxPoolConfig, index)

       case (dropoutConfig: Dropout, index) =>
         new DropoutLayer(dropoutConfig, index)
       case (rnnConfig: RNN, index) =>
         val w = WeightsInitializer.initWeights(
           WeightsInitType.NORMAL,
           Array(rnnConfig.nbInput,
             rnnConfig.nbOutput))

         val wh = WeightsInitializer.initWeights(
           WeightsInitType.NORMAL,
           Array(rnnConfig.nbOutput,
             rnnConfig.nbOutput))

         val h = Nd4j.zeros(rnnConfig.nbOutput)
         val b = Nd4j.zeros(rnnConfig.nbOutput)
         val params = mutable.Map[String, INDArray](
           ParamType.toString(ParamType.W, index) -> w,
           ParamType.toString(ParamType.B, index) -> b,
           ParamType.toString(ParamType.WH, index) -> wh,
           ParamType.toString(ParamType.H, index) -> h)

         new RNNLayer(rnnConfig, params, index)

     }


     val model = new SequentialModel(newLayers)
     val config = ModelConfig(layers, loss, optimizerConfig)
     val params = model.layers
     new Solver(model, config)
   }

}



object Sequential {


  def apply() : Sequential = {
    new Sequential()
  }

  def save(model: Model, file: String) = {
   ModelSerializer.save(model, file)
  }

  def load(file: String) = {
    ModelSerializer.load(file)
  }

}

