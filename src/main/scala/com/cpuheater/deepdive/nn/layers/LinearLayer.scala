package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, ReLU}
import com.cpuheater.deepdive.nn.Linear
import com.cpuheater.deepdive.nn.layers.ParamType.PreOutput
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable

class LinearLayer(layerConfig: Linear,
                  override val params: mutable.Map[String, INDArray],
                  override val layerNb: Int) extends Layer with HasParams{

  private val cache: mutable.Map[String, INDArray] = mutable.Map[String, INDArray]()


  override def name: String = layerConfig.name

  override def activationFn: ActivationFn = layerConfig.activation

  def nbOutput: Int = layerConfig.nbOutput

  def nbInput: Int = layerConfig.nbInput


  override def forward(x: INDArray, isTraining: Boolean =  true): INDArray = {
    val (out, preOutput)= innerForward(x, isTraining)
    out
  }

  private def innerForward(x: INDArray, isTraining: Boolean): (INDArray, INDArray) = {
    val w = params(ParamType.toString(ParamType.W, layerNb))
    val b = params(ParamType.toString(ParamType.B, layerNb))
    val preOutput = x.reshape(x.shape()(0), -1).dot(w).addRowVector(b)
    val out = activationFn(preOutput)
    (out, preOutput)
  }

  override def backward(x: INDArray, dout: INDArray, isTraining: Boolean = true): GradResult = {
    val (_, preOutput) = innerForward(x, isTraining)
    val w = params(ParamType.toString(ParamType.W, layerNb))
    val preOutputDupl = activationFn.derivative(preOutput.dup())
    val da = preOutputDupl * dout
    val dx = da.dot(w.T).reshape(x.shape(): _*)
    val dw = x.reshape(x.shape()(0), -1).T.dot(da)
    val db = da.sum(0)

    GradResult(dx, Map(s"${ParamType.W}${layerNb}" ->dw, s"${ParamType.B}${layerNb}"->db))
  }

  override def toString(): String = s"number of input = ${nbInput} number of output = ${nbOutput}"


}
