package com.cpuheater.deepdive.weights

import com.cpuheater.deepdive.nn.WeightsInitType
import org.apache.commons.math3.util.FastMath
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

object WeightsInitializer {


   def initWeights(`type`: WeightsInitType, nbInput: Int, nbOutput: Int, scale: Double = 1): INDArray = {
       `type` match {
           case WeightsInitType.UNIFORM  =>
             val a = 1.0 / Math.sqrt(nbInput)
             val ret = Nd4j.rand(Array(nbInput, nbOutput)) * scale
             ret
           case WeightsInitType.XAVIER =>
             val ret = Nd4j.randn(Array(nbInput, nbOutput)).muli(FastMath.sqrt(2.0 / (nbInput + nbOutput)))
             ret
       }
   }

}
