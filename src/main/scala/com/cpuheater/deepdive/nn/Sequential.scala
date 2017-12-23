package com.cpuheater.deepdive.nn


import com.cpuheater.deepdive.lossfunctions.{LossFunction, LossFunction2, SoftMaxLoss}
import com.cpuheater.deepdive.nn.layers.{CompType, Layer, LinearLayer}
import com.cpuheater.deepdive.nn.core.{Config, FeedForwardNetwork, SequentialModel, Solver}

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

class Sequential protected (layers: List[LayerConfig] = Nil) {

   def add(layer: LayerConfig)  = new Sequential(layer::layers)

   def build(loss: LossFunction2, lr: Double, batchSize: Int): Solver = {

     Nd4j.getRandom().setSeed(1)

     val l1 = scala.collection.mutable.Map[CompType, INDArray]()
     l1(CompType.W) = Nd4j.create(Array(0.09700684, -0.01866912, -0.16777732, 0.13514824, 0.04932248, -0.13881888, 0.09700684, -0.01866912, -0.16777732)).reshape(3, 3)
     l1(CompType.B) = Nd4j.zeros(3)

     val l2 = scala.collection.mutable.Map[CompType, INDArray]()
     l2(CompType.W) = Nd4j.create(Array(-0.07079329, 0.11830491, .13195695, 0.08975121,  0.13044141,  0.05507294, 0.08975121,  0.13044141,  0.05507294f)).reshape(3, 3)
     l2(CompType.B) = Nd4j.zeros(3)

     val l3 = scala.collection.mutable.Map[CompType, INDArray]()
     l3(CompType.W) = Nd4j.create(Array( 0.03155537,  0.16341055, -0.08454048, 0.10095577,
       0.00948496,  0.13195695, 0.08975121,  0.13044141,  0.05507294, 0.08975121,  0.13044141,  0.05507294, 0.09700684, -0.01866912, -0.16777732,  0.13514824,
       0.04932248, -0.13881888,
       -0.21343547,  0.08187358,  0.0972448 , -0.07079329,
       0.01285232,  0.11055773,0.11830491,  0.01397892, -0.03056505, 0.02692668,0.02095663,  0.04239022)).reshape(3, 10)
     l3(CompType.B) = Nd4j.zeros(10)


     val newLayers = layers.zip(List(l1, l2, l3)).map{
       case (lConfig: Linear, l) =>
         new LinearLayer(lConfig.nbOutput,
           lConfig.nbInput, lConfig.activation, lConfig.name, l)
     }


     val model = new SequentialModel(newLayers)
     val config = Config(layers, loss, lr, batchSize)
     new Solver(model, config)
   }

}



object Sequential {

  def apply() : Sequential = {
    new Sequential()
  }

}

