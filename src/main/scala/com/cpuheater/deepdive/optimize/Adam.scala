package com.cpuheater.deepdive.optimize

import com.cpuheater.deepdive.nn.Optimizer
import org.apache.commons.math3.util.FastMath
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.mutable
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

class Adam(config: Optimizer.Adam, lastGrad: mutable.Map[String, INDArray]) extends BaseOptimizer {

  val mKey = "m"
  val vKey = "v"
  val tKey = "t"
  val keys = lastGrad.keys.toList
  var t: Int = 0

  keys.map{
    key =>
      val a = lastGrad(key)
      lastGrad(s"${mKey}${key}") = a
      lastGrad(s"${vKey}${key}") = a.dup()
      lastGrad(s"${tKey}${key}") = Nd4j.create(Array(1.0f))
      lastGrad.remove(key)
  }


  def optimize(param: INDArray, grad: INDArray, key: String): INDArray = {

    val m = lastGrad(s"${mKey}${key}")
    val v = lastGrad(s"${vKey}${key}")
    val t = lastGrad(s"${tKey}${key}")(0).toInt +1

    m.muli(config.beta1).addi(grad.mul(1.0 - config.beta1))
    v.muli(config.beta2).addi(grad.mul(grad).muli(1 - config.beta2))
    v.muli(config.beta2).addi(grad.mul(grad).muli(1 - config.beta2))

    val mt = m.div(1 - FastMath.pow(config.beta1, t))
    val vt = v.div(1- FastMath.pow(config.beta2, t))

    lastGrad(s"${tKey}${key}") = Nd4j.create(Array(t.toFloat))

    mt.mul(config.lr).div(Transforms.sqrt(vt, true).addi(config.eps))
  }

}

