package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.ActivationFn
import org.nd4j.linalg.api.ndarray.INDArray

import scala.collection.mutable

trait Layer {

  def name: String

  def activationFn: ActivationFn

  def forward(x: INDArray, isTraining: Boolean=true): INDArray

  def backward(x: INDArray, dout: INDArray, isTraining: Boolean=true): GradResult

  def params: mutable.Map[String, INDArray]

}


case class GradResult(dx: INDArray,
                      grads: Map[String, INDArray] = Map.empty[String, INDArray],
                      hidden: Option[INDArray] = None,
                      context: Option[INDArray] = None)




