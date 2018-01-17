package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, ReLU}
import com.cpuheater.deepdive.nn.{Linear, RNN}
import com.cpuheater.deepdive.nn.layers.ParamType.PreOutput
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable

class RNNLayer(layerConfig: RNN,
                  override val params: mutable.Map[String, INDArray],
                  layerNb: Int) extends Layer {

  private val cache: mutable.Map[String, INDArray] = mutable.Map[String, INDArray]()


  override def name: String = layerConfig.name

  override def activationFn: ActivationFn = layerConfig.activation

  def nbOutput: Int = layerConfig.nbOutput

  def nbInput: Int = layerConfig.nbInput


  override def forward(x: INDArray, isTraining: Boolean =  true): INDArray = {

    val w = params(ParamType.toString(ParamType.W, layerNb))
    val b = params(ParamType.toString(ParamType.B, layerNb))
    val wh = params(ParamType.toString(ParamType.WH, layerNb))
    val hidden = params(ParamType.toString(ParamType.H, layerNb))


    val preOutput = (x.dot(w) +  hidden.dot(wh)).addRowVector(b)
    val out = activationFn(preOutput)

    cache(ParamType.toString(ParamType.PreOutput, layerNb)) = preOutput
    cache(ParamType.toString(ParamType.X, layerNb)) = x
    cache(ParamType.toString(ParamType.H, layerNb)) = hidden
    out
  }

  override def backward(dout: INDArray, isTraining: Boolean = true): (INDArray, INDArray, INDArray) = {
    val preOutput = cache(ParamType.toString(ParamType.PreOutput, layerNb))
    val x = cache(ParamType.toString(ParamType.X, layerNb))
    val w = params(ParamType.toString(ParamType.W, layerNb))
    val wh = params(ParamType.toString(ParamType.WH, layerNb))
    val hidden = params(ParamType.toString(ParamType.H, layerNb))
    val b = params(ParamType.toString(ParamType.B, layerNb))

    /**
      * Wx, Wh, x, prev_h, h_raw = cache
    dh_raw = (1 - np.tanh(h_raw) ** 2) * dnext_h
    dx = np.dot(dh_raw, Wx.T)
    dprev_h = np.dot(dh_raw, Wh.T)
    dWx = np.dot(x.T, dh_raw)
    dWh = np.dot(prev_h.T, dh_raw)
    db = np.sum(dh_raw, axis=0)

      *
      */

    val da = activationFn.derivative(preOutput.dup()) * dout

    val dx = da.dot(w.T)
    val dhidden = da.dot(wh.T)
    val dw = x.T.dot(da)
    val dwh = hidden.T.dot(da)
    val db = Nd4j.sum(da, 0)
    (dx, dhidden, dw/*, dwh, db*/)

  }

  override def toString(): String = s"number of input = ${nbInput} number of output = ${nbOutput}"

}
