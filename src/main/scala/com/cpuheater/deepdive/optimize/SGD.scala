package com.cpuheater.deepdive.optimize

import com.cpuheater.deepdive.nn.Optimizer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

class SGD(config: Optimizer.SGD) extends BaseOptimizer {

  def optimize(param: INDArray, grads: INDArray, key: String): INDArray = {
    param - grads * config.lr
  }

}
