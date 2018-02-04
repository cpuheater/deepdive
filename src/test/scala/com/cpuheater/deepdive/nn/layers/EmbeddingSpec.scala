package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, Identity, ReLU}
import com.cpuheater.deepdive.nn.{Conv2d, Dropout, Embedding}
import com.cpuheater.deepdive.util.TestSupport
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer
import org.junit.Assert.assertEquals
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex.point
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex, NDArrayIndexAll}
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._

import scala.collection.mutable



class EmbeddingSpec extends TestSupport {

  it should "test embedding" in {

    Nd4j.setDataType(DataBuffer.Type.DOUBLE)
    Nd4j.getRandom.setSeed(1)

    val (n, t, v, d) = (2, 3, 4, 5)
    val config = Embedding(v, d)
    val x = Nd4j.create(Array(Array(3.0, 0, 0), Array(1.0, 3, 2)))
    val w = Nd4j.create(Array(
      Array( 0.93210596, -0.52653556),
      Array( 1.81533659, -0.35928654),
      Array(-0.45407413, -0.01604144),
      Array(-0.12127493, -0.70861397)))

    val dout = Nd4j.create(
      Array(
        Array(0.62612298, -0.56680738,
              0.41310222, -0.59193812,  0.25920004,  0.90591416),
        Array( 0.17929223,  0.12758901,
          0.97033142, -0.16819401,  0.29146285,  0.02905275))).reshape(2, 3, 2)

    val isTraining = true
    val layerNb = 1
    val params =  mutable.Map[String, INDArray]()
    params(ParamType.toString(ParamType.W, layerNb)) = w
    val layer  = new EmbeddingLayer(config, params, 1)
    val out = layer.forward(x, isTraining)

    ArrayUtil.equals(out.data().asFloat(), Array(-0.12127493, -0.70861397,  0.93210596, -0.52653556,  0.93210596, -0.52653556,
       1.81533659, -0.35928654, -0.12127493, -0.70861397, -0.45407413, -0.01604144)) should be(true)
    val GradResult(_, grads, _, _) = layer.backward(x, dout, isTraining)
    val dw = grads(s"${ParamType.W}${layerNb}")
    ArrayUtil.equals(dw.data().asFloat(), Array(0.67230226,  0.31397604,  0.17929223,  0.12758901,  0.29146285,  0.02905275, 1.59645441, -0.73500139)) should be(true)
  }

}
