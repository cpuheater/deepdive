package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, ReLU}
import com.cpuheater.deepdive.nn.{Embedding, Linear}
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable

class EmbeddingLayer(config: Embedding,
                     override val params: mutable.Map[String, INDArray],
                     layerNb: Int) extends Layer {

  private val cache: mutable.Map[String, INDArray] = mutable.Map[String, INDArray]()

  override def name: String = config.name

  override def activationFn: ActivationFn = throw new UnsupportedOperationException()

  def nbOutput: Int = config.nbOutput

  def nbInput: Int = config.nbInput

  override def forward(x: INDArray, isTraining: Boolean =  true): INDArray = {
    val out= innerForward(x, isTraining)
    out
  }

  private def innerForward(x: INDArray, isTraining: Boolean): INDArray = {
    val w = params(ParamType.toString(ParamType.W, layerNb))

    val xReshaped = x.reshape(x.length(), 1)
    val indexes = (0 until xReshaped.length()).map( i => xReshaped.getInt(i, 0)).toArray
    val rows = Nd4j.pullRows(w, 1, indexes)
    rows
  }

  override def backward(x: INDArray, dout: INDArray, isTraining: Boolean = true): GradResult = {

    val w = params(ParamType.toString(ParamType.W, layerNb))
    val dw = Nd4j.zerosLike(w)
    val Array(n, t, v) = dout.shape()
    val doutReshaped = dout.reshape(n*t,v)
    val xReshaped = x.reshape(x.length(), 1)
    (0 until xReshaped.length()).foreach{
      i =>
        val ddd = doutReshaped.getRow(i)
        dw.getRow(xReshaped.getInt(i, 0)).addi(doutReshaped.getRow(i))
    }
    GradResult(dw, Map(s"${ParamType.W}${layerNb}" ->dw))
  }

  override def toString(): String = s"number of input = ${nbInput} number of output = ${nbOutput}"

}
