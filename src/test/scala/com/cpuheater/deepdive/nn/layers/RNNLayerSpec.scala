package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{ActivationFn, Identity, Tanh}
import com.cpuheater.deepdive.nn.layers.{GradResult, ParamType, RNNLayer}
import com.cpuheater.deepdive.util.TestSupport
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._

import scala.collection.mutable



class RNNLayerSpec extends TestSupport {

  it should "RNN" in {

    /***
      *
      * x = np.ones((N, T* D)).reshape(N, T, D) + 0.1
h0 = np.random.randn(N, H)
Wx = np.random.randn(D, H)
Wh = np.random.randn(H, H)
b = np.random.randn(H)
      *
      */

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

    val GradResult(dx, grads, Some(currentHidden), _) = rnn.backward(x, dout)
    val dw = grads(s"${ParamType.W}${layerNb}")
    val db = grads(s"${ParamType.B}${layerNb}")
    val dwh = grads(s"${ParamType.WH}${layerNb}")

    ArrayUtil.equals(dwh.data().asFloat(), Array(1.447644966854219E-5,3.52865020510127E-9,0.0015752806224597095,-9.825187852343842E-6,-2.3949000740629612E-9,-0.0010484676037250194,-1.5193237582146287E-5,-3.703368039397262E-9,-0.0016564911061348513)) should be(true)
    ArrayUtil.equals(db.data().asFloat(), Array(0.04156935560000275,0.8177794829228611,-0.018020760596494295)) should be(true)
    ArrayUtil.equals(dw.data().asFloat(), Array(-4.578259345591776E-5,-1.1156477654624965E-8,-0.01134820371895751,0.041523573006546834,0.8177794717663834,-0.029368964315451804,0.08309292860654958,1.6355589546892446,-0.047389724911946096)) should be(true)
    ArrayUtil.equals(currentHidden.data().asFloat(), Array(-0.11284599694988041,-9.99657616275682E-4,0.8151347920307274,-3.949800798413231E-5,0.11428025476483286,-0.0010125643834864194)) should be(true)



  }


}
