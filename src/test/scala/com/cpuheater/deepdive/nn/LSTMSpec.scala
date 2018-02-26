package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.util.TestSupport
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._



class LSTMSpec extends TestSupport {

  private case class FFData(x: INDArray, out: INDArray, preOutput: INDArray, prevHidden: INDArray, w: INDArray, wh: INDArray)

  it should "lstm" in {

    /**
      * N, D, H = 4, 5, 6
x = np.random.randn(N, D)
prev_h = np.random.randn(N, H)
prev_c = np.random.randn(N, H)
Wx = np.random.randn(D, 4 * H)
Wh = np.random.randn(H, 4 * H)
b = np.random.randn(4 * H)
      */

    val (n, d, h) = (3,4,5)
    val x = Nd4j.linspace(-0.4, 1.2, n*d).reshape(n, d)
    val prevH = Nd4j.linspace(-0.3, 0.7, n*h).reshape(n, h)
    val prevC = Nd4j.linspace(-0.4, 0.9, n*h).reshape(n, h)
    val wx = Nd4j.linspace(-2.1, 1.3, 4*d*h).reshape(d, 4*h)
    val wh = Nd4j.linspace(-0.7, 2.2, 4*h*h).reshape(h, 4*h)
    val b = Nd4j.linspace(0.3, 0.7, 4*h)
    val (nextH, nextC, (a, i, f, o, g)) = preOutputStep(x, prevH, prevC, wx, wh, b)

    val dnextH = Nd4j.create(Array(-1.56684685,  0.78359698, -0.61346944, -0.74250482,  0.59537819,
                                   1.19415458, -0.59293609,  1.49635533,  0.34119439,  0.71495757,
                                   -0.41839878, -1.2747691 , -0.50256531, -0.41858449, -0.20158235)).reshape(3, 5)


    val dnextC = Nd4j.create(Array( 0.60837195,  0.42158052,  0.2238923,   0.5886588 ,  0.47241413,
                                    1.43034462, -0.32831242, -1.1131782,   0.91534937, -0.55090836,
                                    0.92657081, -0.49621266, -0.6833397,   0.36247805, -0.78613309)).reshape(3, 5)

    val (dx, dprevH, dprevC, dwx, dwh, db) = backpropStep(dnextH, dnextC, h, x, wx, wh, a, i, f, o, g, prevC, nextC, prevH)


    ArrayUtil.equals(dx.data().asFloat(), Array(-0.69906442, -0.27328623,  0.15249196,  0.57827015,
    -1.31395103, -0.71064064, -0.10733025,  0.49598015,
     1.44172842,  0.77368226,  0.1056361,  -0.56241005))

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

    val cache = (a, i, f, o, g)
    (nextH, nextC, cache)

  }

  def backpropStep(dnextH: INDArray, dnextC: INDArray, h: Int, x: INDArray,
                   wx: INDArray, wh: INDArray, a: INDArray, i: INDArray, f: INDArray, o: INDArray, g: INDArray,
                   preC: INDArray, nextC: INDArray, prevH: INDArray) = {
    val dout = dnextH * tanh(nextC)
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
    val dag = dg * pow(tanh(ag), 2).rsub(1)
    val da = Nd4j.hstack(dai, daf, dao, dag)
    val dx = da.mmul(wx.T)
    val dprevH = da.mmul(wh.T)
    val db = da.sum(0)
    val dwx = x.T.mmul(da)
    val dwh = prevH.T.mmul(da)
    (dx, dprevH, dprevC, dwx, dwh, db)
  }




}
