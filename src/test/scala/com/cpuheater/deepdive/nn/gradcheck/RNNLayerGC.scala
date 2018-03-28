package com.cpuheater.deepdive.nn.gradcheck

import com.cpuheater.deepdive.activations.{ActivationFn, Identity, Tanh}
import com.cpuheater.deepdive.nn.RNN
import com.cpuheater.deepdive.nn.layers.{GradResult, ParamType, RNNLayer}
import com.cpuheater.deepdive.util.{GradientChecking, TestSupport}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._

import scala.collection.mutable



class RNNLayerGCSpec extends TestSupport {

  it should "RNN" in {

    Nd4j.setDataType(DataBuffer.Type.DOUBLE)

    val (n, d, t, h) = (2,3,2, 3)
    val x = Nd4j.linspace(0, 11, t*d*n).reshape(n, t, d)
    val hidden = Nd4j.zeros(n, h)
    val wx = Nd4j.create(Array(Array(-0.1497979,   1.86172902, -1.4255293d ),
      Array(-0.3763567,  -0.34227539 , 0.29490764d),
      Array(-0.83732373 , 0.95218767,  1.32931659)))



    val wh = Nd4j.create(Array(Array( 0.52465245, -0.14809998,  0.88953195),
      Array( 0.12444653 , 0.99109251,  0.03514666),
      Array( 0.26207083 , 0.14320173,  0.90101716)))

    val b = Nd4j.create(Array( 0.23185863, -0.79725793d,  0.12001014))

    val layerNb = 1

    val params =  mutable.Map[String, INDArray]()
    params(ParamType.toString(ParamType.W, layerNb)) = wx
    params(ParamType.toString(ParamType.B, layerNb)) = b
    params(ParamType.toString(ParamType.WH, layerNb)) = wh
    params(ParamType.toString(ParamType.H, layerNb)) = hidden

    val rnn = new RNNLayer(RNN(nbInput = d, nbOutput = h, activation = Tanh), params, layerNb)

    val out = rnn.forward(x)


    val dout = Nd4j.create(Array( 0.41794341,  1.39710028, -1.78590431, -0.70882773, -0.07472532, -0.77501677,
      -0.1497979,   1.86172902 ,-1.4255293 , -0.3763567 , -0.34227539 , 0.29490764)).reshape(n, t, d)


    val fx = (x: INDArray) => rnn.forward(x)
    val fwx = (wx: INDArray) => {
      params(ParamType.toString(ParamType.W, layerNb)) = wx
      val rnn = new RNNLayer(RNN(nbInput = d, nbOutput = h, activation = Tanh), params, layerNb)
      rnn.forward(x)
    }

    val fwh = (wh: INDArray) => {
      params(ParamType.toString(ParamType.WH, layerNb)) = wh
      val rnn = new RNNLayer(RNN(nbInput = d, nbOutput = h, activation = Tanh), params, layerNb)
      rnn.forward(x)
    }

    val fb = (b: INDArray) => {
      params(ParamType.toString(ParamType.B, layerNb)) = b
      val rnn = new RNNLayer(RNN(nbInput = d, nbOutput = h, activation = Tanh), params, layerNb)
      rnn.forward(x)
    }

    val dxNG = GradientChecking.check(x, fx, dout)
    val dwxNG = GradientChecking.check(wx, fwx, dout)
    val dwhNG = GradientChecking.check(wh, fwh, dout)
    val dbNG = GradientChecking.check(b, fb, dout)


    val GradResult(dx, grads, Some(currentHidden), _) = rnn.backward(x, dout)
    val dw = grads(s"${ParamType.W}${layerNb}")
    val db = grads(s"${ParamType.B}${layerNb}")
    val dwh = grads(s"${ParamType.WH}${layerNb}")


    println(GradientChecking.error(dx, dxNG))
    println(GradientChecking.error(dw, dwxNG))
    println(GradientChecking.error(dwh, dwhNG))
    println(GradientChecking.error(db, dbNG))

  }


}
