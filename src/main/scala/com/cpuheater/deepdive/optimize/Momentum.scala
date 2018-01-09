package com.cpuheater.deepdive.optimize

import com.cpuheater.deepdive.nn.Optimizer
import org.nd4j.linalg.api.ndarray.INDArray
import scala.collection.mutable
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

class Momentum(config: Optimizer.Momentum, v: mutable.Map[String, INDArray]) extends BaseOptimizer {

  def optimize(param: INDArray, grad: INDArray, key: String): INDArray = {
    v(key) = v(key) * config.momentum + grad * config.lr
    v(key)
  }

}

