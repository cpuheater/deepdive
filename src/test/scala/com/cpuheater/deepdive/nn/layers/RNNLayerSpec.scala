package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{ActivationFn, Identity, Tanh}
import com.cpuheater.deepdive.nn.layers.{ParamType, RNNLayer}
import com.cpuheater.deepdive.util.TestSupport
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable



class RNNLayerSpec extends TestSupport {

  it should "lstm" in {


    /**
      *
      * N, D, H = 4, 5, 6
x = np.random.randn(N, D)
h = np.random.randn(N, H)
Wx = np.random.randn(D, H)
Wh = np.random.randn(H, H)
b = np.random.randn(H)
      *
      *
      */

    val (n, d, h) = (4,2,3)
    val x = Nd4j.linspace(-0.4, 1.2, n*d).reshape(n, d)
    val hidden = Nd4j.linspace(-0.3, 0.7, n*h).reshape(n, h)
    val wx = Nd4j.linspace(-2.1, 1.3, d*h).reshape(d, h)
    val wh = Nd4j.linspace(-0.7, 2.2, h*h).reshape(h, h)
    val b = Nd4j.linspace(0.3, 0.7, h)

    val layerNb = 1

    val params =  mutable.Map[String, INDArray]()
    params(ParamType.toString(ParamType.W, layerNb)) = wx
    params(ParamType.toString(ParamType.B, layerNb)) = b
    params(ParamType.toString(ParamType.WH, layerNb)) = wh
    params(ParamType.toString(ParamType.H, layerNb)) = hidden

    val rnn = new RNNLayer(RNN(nbInput = d, nbOutput = h, activation = Tanh), params, layerNb)

    val out = rnn.forward(x)

    val (_, _, _) = rnn.backward(out)


  }


  def preOutputStep(x: INDArray, prevH: INDArray,
                    prevC: INDArray, wx: INDArray,
                    wh: INDArray, b: INDArray) = {


    /**
      *
      * wFFTranspose = recurrentWeights
                            .get(NDArrayIndex.all(), interval(4 * hiddenLayerSize, 4 * hiddenLayerSize + 1))
                            .transpose(); //current
            wOOTranspose = recurrentWeights
                            .get(NDArrayIndex.all(), interval(4 * hiddenLayerSize + 1, 4 * hiddenLayerSize + 2))
                            .transpose(); //current
            wGGTranspose = recurrentWeights
                            .get(NDArrayIndex.all(), interval(4 * hiddenLayerSize + 2, 4 * hiddenLayerSize + 3))
                            .transpose(); //previous

      */

    val h = prevH.size(1)
    val a = (x.mmul(wx) + prevH.mmul(wh)).addiRowVector(b)
    val ai = a.get(NDArrayIndex.all(), NDArrayIndex.interval(0*h, 1*h))
    val af = a.get(NDArrayIndex.all(), NDArrayIndex.interval(1*h, 2*h))
    val ao = a.get(NDArrayIndex.all(), NDArrayIndex.interval(2*h, 3*h))
    val ag = a.get(NDArrayIndex.all(), NDArrayIndex.interval(3*h, 4*h))
    val i = sigmoid(ai)
    val f = sigmoid(af)
    val o = sigmoid(ao)
    val g = tanh(ag)
    val nextC = f * prevC + i * g
    val nextH = o * tanh(nextC)

    val cache = (h, x, wx, wh, a, i, f, o, g, prevC, prevH, nextC, prevH)
    (nextH, nextC, cache)

  }

  def backpropStep(dnextH: INDArray, dnextC: INDArray, h: Int, x: INDArray,
                   wx: INDArray, wh: INDArray, a: INDArray, i: INDArray, f: INDArray, o: INDArray, g: INDArray,
                   preC: INDArray, nextC: INDArray, prevH: INDArray) = {
    val dout = dnextC * tanh(nextC)
    val dc = (pow(tanh(nextC), 2).rsub(1)) * dnextH*o + dnextC
    val df = dc * preC
    val dprevC = dc *f
    val di = g*dc
    val dg = i * dc
    val ai = a.get(NDArrayIndex.all(), NDArrayIndex.interval(0*h, 1*h))
    val af = a.get(NDArrayIndex.all(), NDArrayIndex.interval(1*h, 2*h))
    val ao = a.get(NDArrayIndex.all(), NDArrayIndex.interval(2*h, 3*h))
    val ag = a.get(NDArrayIndex.all(), NDArrayIndex.interval(3*h, 4*h))
    val dai = di * (sigmoid(ai).rsub(1)) * sigmoid(ai)
    val daf = df * (sigmoid(af).rsub(1)) * sigmoid(af)
    val dao = dout * (sigmoid(ao).rsub(1)) * sigmoid(ao)
    val dag = dout * (sigmoid(ao).rsub(1)) * sigmoid(ao)
    val da = Nd4j.hstack(dai, daf, dao, dag)
    val dx = da.mmul(wx.T)
    val dprevH = da.mmul(wh.T)
    val db = da.sum(0)
    val dwx = x.T.mmul(da)
    val dwh = prevH.T.mmul(da)
    (dx, dprevH, dprevC, dwx, dwh, db)
  }




}
