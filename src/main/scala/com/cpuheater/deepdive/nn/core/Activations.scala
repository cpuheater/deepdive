/*
package com.cpuheater.deepdive.nn.core

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

trait Activation extends Function1[INDArray, INDArray] {

  def apply(arg: INDArray): INDArray

  def derivative(arg: INDArray): INDArray

}


object Activation {

  object Sigmoid extends Activation {

    def apply(a: INDArray): INDArray = sigmoid(a)

    def derivative(x: INDArray): INDArray = {
      x*(x.rsub(1))
    }
  }

  object Identity extends Activation {
    def apply(a: INDArray): INDArray = a

    def derivative(x: INDArray): INDArray = {
      Nd4j.onesLike(x)
    }
  }

  object Relu extends Activation {
    def apply(a: INDArray): INDArray = Transforms.relu(a)

    def derivative(x: INDArray): INDArray =
      Nd4j.getExecutioner.execAndReturn(new RectifedLinear(x).derivative)
  }

}




*/
