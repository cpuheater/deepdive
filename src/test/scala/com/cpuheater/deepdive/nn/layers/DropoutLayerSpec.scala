package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, Identity, ReLU}
import com.cpuheater.deepdive.nn.{Conv2d, Dropout}
import com.cpuheater.deepdive.util.TestSupport
import org.junit.Assert.assertEquals
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex.point
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex, NDArrayIndexAll}
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._

import scala.collection.mutable



class DropoutLayerSpec extends TestSupport {

  it should "test dropout" in {

    val dropOut = 0.9
    Nd4j.getRandom.setSeed(1)
    val config = Dropout(dropOut, name = "")
    val dout = Nd4j.randn(2, 5)
    val x = Nd4j.randn(2, 5)

    val isTraining = true

    val layer  = new DropoutLayer(config = config, 1)

    val out = layer.forward(x, isTraining)

    ArrayUtil.equals(out.data().asFloat(), Array(-1.9587543,0.40646845,1.2421689,1.0452389,0.22673291,-0.35432237,0.31878293,0.5981538,-0.6759827,-0.524728)) should be(true)

    val GradResult(dx, _) = layer.backward(x, dout, isTraining)

    ArrayUtil.equals(dx.data().asFloat(), Array(-0.5451878,-0.72581875,1.1784248,-0.05978557,1.1411147,-0.86540264,0.3228806,0.10490196,0.58895004,-0.89353514)) should be(true)
  }

}
