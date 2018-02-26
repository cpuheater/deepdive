package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{ActivationFn, Identity, Tanh}
import com.cpuheater.deepdive.nn.layers.{GradResult, LSTMLayer, ParamType, RNNLayer}
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



class LSTMLayerSpec extends TestSupport {

  it should "LSTM" in {

    Nd4j.setDataType(DataBuffer.Type.DOUBLE)

    val (n, d, t, h) = (2,3,2, 3)


    val x = Nd4j.create(Array( 0.41794341,  1.39710028, -1.78590431,
    -0.70882773, -0.07472532, -0.77501677, -0.1497979,   1.86172902, -1.4255293,
    -0.3763567,  -0.34227539,  0.29490764)).reshape(n, t, d)



    val hidden = Nd4j.create(Array(-0.83732373,  0.95218767,  1.32931659,
      0.52465245, -0.14809998,  0.88953195)).reshape(n, h)

    val wx = Nd4j.create(Array( 0.12444653,  0.99109251,  0.03514666,  0.26207083,  0.14320173,  0.90101716,
    0.23185863, -0.79725793,  0.12001014, -0.65679608,  0.26917456,  0.333667,
     0.27423503,  0.76215717, -0.69550058,  0.29214712, -0.38489942,  0.1228747,
    -1.42904497,  0.70286283, -0.85850947, -1.14042979, -1.58535997, -0.01530138,
    -0.32156083,  0.56834936, -0.19961722,  1.27286625,  1.27292534,  1.58102968,
    -1.75626715, 0.9217743,  -0.6753054,  -1.43443616,  0.47021125,  0.03196734))
      .reshape(d, 4*h)



    val wh = Nd4j.create(Array(
      0.04448574,  0.47824879, -2.51335181, -1.15740245, -0.70470413, -1.04978879,
    -1.90795589,  0.49258765,  0.83736166, -1.4288134,  -0.18982427, -1.14094943,
    -2.12570755, -0.41354791,  0.44148975,  0.16411113, -0.65505065, -0.30212765,
    -0.25704466, -0.12841368,  0.26338593,  0.1672181,  -0.30871951, -1.26754462,
    -0.22319022, -0.82993433, -1.11271826, -0.44613095, -0.40001719,  0.36343905,
    0.94992777, -0.32379447,  0.27031704, -0.63381148, -2.71484268,  0.65576139)).reshape(d, 4 * h)

    val b = Nd4j.create(Array(-1.17004858,  0.0598685,  -1.64182729, -0.28069634, -0.67946972, -1.80480094,
    0.53770564, -0.12171369, -1.04250949,  0.13828792, -0.22557183, -1.1928829))

    val layerNb = 1

    val params =  mutable.Map[String, INDArray]()
    params(ParamType.toString(ParamType.W, layerNb)) = wx
    params(ParamType.toString(ParamType.B, layerNb)) = b
    params(ParamType.toString(ParamType.WH, layerNb)) = wh
    params(ParamType.toString(ParamType.H, layerNb)) = hidden

    val lstm = new LSTMLayer(LSTM(nbInput = d, nbOutput = h, activation = Tanh), params, layerNb)

    val out = lstm.forward(x)


    val dout = Nd4j.create(Array(
      -0.68320696,  0.19909408,  0.03070661,
    -0.44972639,  0.14447532, -0.35229594,
     0.4882136,  -0.4347099,  -0.28692265,
    -0.84338097, -0.10827394,  0.85434757 )).reshape(out.shape():_*)

    val GradResult(dx, grads, Some(currentHidden), _) = lstm.backward(x, dout)
    val dw = grads(s"${ParamType.W}${layerNb}")
    val db = grads(s"${ParamType.B}${layerNb}")
    val dwh = grads(s"${ParamType.WH}${layerNb}")

    ArrayUtil.equals(dwh.data().asFloat(), Array(1.447644966854219E-5,3.52865020510127E-9,0.0015752806224597095,-9.825187852343842E-6,-2.3949000740629612E-9,-0.0010484676037250194,-1.5193237582146287E-5,-3.703368039397262E-9,-0.0016564911061348513)) should be(true)
    ArrayUtil.equals(db.data().asFloat(), Array(0.04156935560000275,0.8177794829228611,-0.018020760596494295)) should be(true)
    ArrayUtil.equals(dw.data().asFloat(), Array(-4.578259345591776E-5,-1.1156477654624965E-8,-0.01134820371895751,0.041523573006546834,0.8177794717663834,-0.029368964315451804,0.08309292860654958,1.6355589546892446,-0.047389724911946096)) should be(true)
    ArrayUtil.equals(currentHidden.data().asFloat(), Array(-0.11284599694988041,-9.99657616275682E-4,0.8151347920307274,-3.949800798413231E-5,0.11428025476483286,-0.0010125643834864194)) should be(true)



  }


}
