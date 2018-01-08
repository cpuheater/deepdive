package com.cpuheater.deepdive.optimize

import com.cpuheater.deepdive.nn.Optimizer
import org.nd4j.linalg.api.ndarray.INDArray

import scala.collection.mutable

class Momentum(conf: Optimizer.Momentum, params: mutable.Map[String, INDArray]) extends BaseOptimizer {

  def optimize(param: INDArray, grad: INDArray, key: String): INDArray = {
    /*
    velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            model[layer] += velocity[layer]
     */
    ???
  }

}

