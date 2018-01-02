package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.ActivationFn
import org.nd4j.linalg.api.ndarray.INDArray

import scala.collection.mutable

trait Layer {

  def name: String

  def activationFn: ActivationFn

  def forward(x: INDArray, isTraining: Boolean=true): INDArray

  def backward(x: INDArray, isTraining: Boolean=true): (INDArray, INDArray, INDArray)

  def params: mutable.Map[String, INDArray]

}





