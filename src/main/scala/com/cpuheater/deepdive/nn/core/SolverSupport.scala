package com.cpuheater.deepdive.nn.core

import com.cpuheater.deepdive.nn.Optimizer
import com.cpuheater.deepdive.optimize.{BaseOptimizer, Momentum, RMSProp, SGD}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable

trait SolverSupport {

  protected def buildOptimizer(config: BuildConfig, params: Map[String, INDArray] ): BaseOptimizer = config.optimizer match {
    case config: Optimizer.SGD =>
      new SGD(config)
    case config: Optimizer.Momentum =>
      new Momentum(config, createParams(Map(params.toSeq: _*)))
    case config: Optimizer.RMSProp =>
      new RMSProp(config, createParams(Map(params.toSeq: _*)))
  }


  private def createParams(params: Map[String, INDArray]): mutable.Map[String, INDArray] = {
    val zerosParams = params.map{ case (key, param) =>  (key -> Nd4j.zerosLike(param))}
    mutable.Map(zerosParams.toSeq: _*)
  }

}
