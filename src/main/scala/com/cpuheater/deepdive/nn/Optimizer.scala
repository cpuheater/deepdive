package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.optimize.{BaseOptimizer, SGD}



trait Optimizer {
  def lr: Double
}

object Optimizer {

  case class SGD(override val lr: Double = 0.01) extends Optimizer
  case class Momentum(override val lr: Double, momentum: Double) extends Optimizer

}



