package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, ReLU}
import com.cpuheater.deepdive.nn.{Dropout, Linear}
import com.cpuheater.deepdive.nn.layers.ParamType.PreOutput
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.random.impl.BinomialDistribution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable

class DropoutLayer(config: Dropout,
                  layerNb: Int) extends Layer {

  private var mask: INDArray = _


  override def name: String = config.name

  override def activationFn: ActivationFn = throw new UnsupportedOperationException()

  def nbOutput: Int = throw new UnsupportedOperationException()

  def nbInput: Int = throw new UnsupportedOperationException()

  def params: mutable.Map[String, INDArray] = throw new UnsupportedOperationException()


  override def forward(x: INDArray, isTraining: Boolean =  true): INDArray = {
    if(isTraining){
      mask = Nd4j.getExecutioner.exec(new BinomialDistribution(Nd4j.zeros(x.shape(): _*), 1, config.dropOut))
      val out = x * mask/config.dropOut
      out
    } else {
      x
    }

  }

  override def backward(x:INDArray, dout: INDArray, isTraining: Boolean = true): GradResult = {
    if(isTraining){
      val out = dout * mask
      GradResult(out)
    } else {
      GradResult(dout)
    }

  }

  override def toString(): String = s"number of input = ${nbInput} number of output = ${nbOutput}"


}
