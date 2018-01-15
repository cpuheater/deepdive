package com.cpuheater.deepdive.optimize

import com.cpuheater.deepdive.nn.Optimizer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.mutable
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

class RMSProp(config: Optimizer.RMSProp, lastGrad: mutable.Map[String, INDArray]) extends BaseOptimizer {

  def optimize(param: INDArray, grad: INDArray, key: String): INDArray = {
    lastGrad(key) = lastGrad(key).muli(config.decay).addi(grad.mul(grad).muli(1 - config.decay))
    grad.muli(config.lr).divi(Transforms.sqrt(lastGrad(key).dup, false).addi(config.eps))
  }

}

