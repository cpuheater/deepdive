package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, Identity}
import com.cpuheater.deepdive.nn.Conv
import com.cpuheater.deepdive.nn.layers.Convolutional
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



class ConvLayerSpec extends TestSupport {

  it should "linspace" in {

    val batchSize = 2
    val height = 4
    val width = 4
    val channels = 3
    val nbOfFilters = 3
    val filterHeight = 4
    val filterWidth = 4
    val stride = 2
    val padding = 1

    val config = Conv(height = height,
      width = width,
      channels = channels,
      nbOfFilters = nbOfFilters,
      filterHeight = filterHeight,
      filterWidth = filterWidth,
      stride = stride,
      padding = padding,
      activation = Identity,
      name = "")

    val layerNb = 1

    val outHeight = config.outWidth
    val outWidth = config.outWidth

    val x =  Nd4j.linspace(-0.1, 0.5, 2*3*4*4).reshape(batchSize, channels, height, width)
    val w =  Nd4j.linspace(-0.2, 0.3, 3*3*4*4).reshape(nbOfFilters, channels, filterHeight, filterWidth)
    val b = Nd4j.linspace(-0.1, 0.2, nbOfFilters)
    val dout = Nd4j.linspace(-0.1, 0.2, 24).reshape(2, 3, 2, 2)


    val params =  mutable.Map[String, INDArray]()
    params(CompType.print(CompType.W, layerNb)) = w
    params(CompType.print(CompType.B, layerNb)) = b
    val layer  = new ConvLayer(config = config, params, 1)

    val out = layer.forward(x)

    ArrayUtil.equals(out.data().asFloat(), Array(-0.087598115, -0.9805359, -0.109877825, -1.0314355, -0.18387194, -1.191289, -0.21092162, -1.2469586, 0.21027091, 0.6910836, 0.21661097, 0.6688039, 0.22847627, 0.5948098, 0.23004639, 0.56776005, 0.5081399, 2.3627033, 0.54309976, 2.3690434, 0.64082444, 2.3809085, 0.6710144, 2.3824787)) should be(true)

    val (dx, dw, db) = layer.backward(dout)

  }


}
