package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, ReLU}
import com.cpuheater.deepdive.nn.Linear
import com.cpuheater.deepdive.nn.layers.CompType.PreOutput
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable

class LinearLayer(layerConfig: Linear,
                  override val params: mutable.Map[CompType, INDArray]) extends Layer {

  private val cache: mutable.Map[CompType, INDArray] = mutable.Map[CompType, INDArray]()


  override def name: String = layerConfig.name

  override def activationFn: ActivationFn = layerConfig.activation

  def nbOutput: Int = layerConfig.nbOutput

  def nbInput: Int = layerConfig.nbInput


  override def forward(x: INDArray, isTraining: Boolean =  true): INDArray = {
    val w = params(CompType.W)
    val b = params(CompType.B)
    val preOutput = x.reshape(x.shape()(0), -1).dot(w).addRowVector(b)
    val out = activationFn(preOutput)
    cache(CompType.PreOutput) = preOutput
    cache(CompType.X) = x
    out
  }

  override def backward(dout: INDArray, isTraining: Boolean = true): (INDArray, INDArray, INDArray) = {
    val preOutput = cache(CompType.PreOutput)
    val x = cache(CompType.X)
    val w = params(CompType.W)
    val b = params(CompType.B)

    val preOutputDupl = activationFn.derivative(preOutput.dup())
    val da = preOutputDupl * dout

    val dx = da.dot(w.T).reshape(x.shape(): _*)
    val dw = x.reshape(x.rows(), -1).T.dot(da)
    val db = da.sum(0)
    (dx, dw, db)

  }


}
