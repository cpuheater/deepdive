package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.optimize.{BaseOptimizer, SGD}



trait Optimizer {
  def lr: Double
}

object Optimizer {

  case class SGD(override val lr: Double = 0.01) extends Optimizer
  case class Momentum(override val lr: Double = 0.01,
                      momentum: Double = 0.09) extends Optimizer
  case class RMSProp(override val lr: Double = 0.01,
                     decay: Double =0.99,
                     eps: Double = 1e-8) extends Optimizer

  case class Adam(override val lr: Double = 0.01,
                     beta1: Double =0.9, beta2: Double= 0.999,
                     eps: Double = 1e-8) extends Optimizer

}



