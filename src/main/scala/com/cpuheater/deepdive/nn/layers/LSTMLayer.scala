package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, ReLU}
import com.cpuheater.deepdive.nn.{LSTM, Linear, RNN}
import com.cpuheater.deepdive.nn.layers.ParamType.PreOutput
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{BooleanIndexing, NDArrayIndex}
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

class LSTMLayer(layerConfig: LSTM,
               override val params: mutable.Map[String, INDArray],
                override val layerNb: Int) extends Layer with HasParams {

  private val cache: mutable.Map[String, INDArray] = mutable.Map[String, INDArray]()
  private val cacheTS: mutable.ListBuffer[Map[String, INDArray]] = new ListBuffer[Map[String, INDArray]]()

  private case class FFData(x: INDArray, w: INDArray, wh: INDArray,  a: INDArray, i: INDArray, f: INDArray, o: INDArray,
                            g: INDArray, prevC: INDArray, prevH: INDArray, nextC: INDArray, nextH: INDArray)

  override def name: String = layerConfig.name

  override def activationFn: ActivationFn = layerConfig.activation

  def nbOutput: Int = layerConfig.nbOutput

  def nbInput: Int = layerConfig.nbInput


  override def forward(x: INDArray, isTraining: Boolean =  true): INDArray = {
    val (out, _)  = innerForward(x, isTraining)
    out
  }

  private def innerForward(x: INDArray, isTraining: Boolean =  true): (INDArray, List[FFData]) = {
    val w = params(ParamType.toString(ParamType.W, layerNb))
    val b = params(ParamType.toString(ParamType.B, layerNb))
    val wh = params(ParamType.toString(ParamType.WH, layerNb))
    val prevH = params(ParamType.toString(ParamType.H, layerNb))

    val Array(n, t, d) = x.shape()
    val h = wh.shape()(0)

    val allOut = Nd4j.zeros(Array(n, t, h):_*)

    val allData = ListBuffer[FFData]()
    (0 until t).foreach{
      time =>
        val currentX = x.tensorAlongDimension(time, 0, 2)
        if(time == 0){
          val data = forwardTimeStep(currentX, prevH, Nd4j.zerosLike(prevH), w, wh, b)
          allData += data
          allOut.tensorAlongDimension(time, 0, 2).assign(data.nextH)
        } else {
          val prevH = allOut.tensorAlongDimension(time-1, 0, 2)
          val prevC = allData.last.nextC
          val data = forwardTimeStep(currentX, prevH, prevC, w, wh, b)
          allData +=data
          allOut.tensorAlongDimension(time, 0, 2).assign(data.nextH)
        }
    }

    (allOut, allData.toList)
  }

  private def forwardTimeStep(x: INDArray,
                              prevH: INDArray,
                              prevC: INDArray,
                              w: INDArray,
                              wh: INDArray,
                              b: INDArray): FFData = {

    val h = prevH.size(1)
    val a = (x.mmul(w) + prevH.mmul(wh)).addiRowVector(b)
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

    FFData(x, w, wh,  a, i, f, o, g, prevC, prevH, nextC, nextH)
  }

  def backward(x: INDArray, dout: INDArray, isTraining: Boolean = true): GradResult = {

    val (_, allData) = innerForward(x, isTraining)

    val w = params(ParamType.toString(ParamType.W, layerNb))
    val b = params(ParamType.toString(ParamType.B, layerNb))
    val wh = params(ParamType.toString(ParamType.WH, layerNb))
    var prevHidden = params(ParamType.toString(ParamType.H, layerNb))
    var prevContext = Nd4j.zerosLike(prevHidden)

    val Array(n, t, d) = dout.shape()
    val h = wh.shape()(0)

    var dx = Nd4j.zeros(Array(n, t, d): _*)
    var dw =  Nd4j.zeros(d, 4*h)
    var dwh = Nd4j.zeros(h, 4*h)
    var db = Nd4j.zeros(4*h)

    (t-1 to 0).by(-1).foreach {
      time =>
        val doutTS = dout.tensorAlongDimension(time, 0, 2) + prevHidden

        val (dxTS, dprevHTS, dprevCTS, dwTS, dwhTS, dbTS) = backwardTimeStep(doutTS, prevContext, allData(time))
        dx.tensorAlongDimension(time, 0, 2).assign(dxTS)
        dw = dw + dwTS
        dwh = dwh + dwhTS
        db = db + dbTS
        prevHidden = dprevHTS
        prevContext = dprevCTS
    }

    val grads = Map(s"${ParamType.W}${layerNb}" ->dw,
      s"${ParamType.B}${layerNb}"->db,
      s"${ParamType.WH}${layerNb}"->dwh)
    GradResult(dx, grads, hidden = Some(prevHidden))

  }


  private def backwardTimeStep(dnextH: INDArray, dnextC: INDArray,
                               data: FFData,
                               isTraining: Boolean = true
                              ): (INDArray, INDArray, INDArray, INDArray, INDArray, INDArray) = {

    /**
      * (x: INDArray, w: INDArray, wh: INDArray,  a: INDArray, i: INDArray, f: INDArray, o: INDArray,
                            g: INDArray, prevC: INDArray, prevH: INDArray, nextC: INDArray, nextH: INDArray)
      */
    val h = data.prevH.size(1)
    val dout = dnextH * tanh(data.nextC)
    val dc = pow(tanh(data.nextC), 2).rsub(1) * dnextH*data.o + dnextC
    val df = dc * data.nextC
    val dprevC = dc *data.f
    val di = data.g*dc
    val dg = data.i * dc
    val ai = data.a.get(NDArrayIndex.all(), NDArrayIndex.interval(0*h, 1*h))
    val af = data.a.get(NDArrayIndex.all(), NDArrayIndex.interval(1*h, 2*h))
    val ao = data.a.get(NDArrayIndex.all(), NDArrayIndex.interval(2*h, 3*h))
    val ag = data.a.get(NDArrayIndex.all(), NDArrayIndex.interval(3*h, 4*h))
    val dai = di * sigmoid(ai).rsub(1) * sigmoid(ai)
    val daf = df * sigmoid(af).rsub(1) * sigmoid(af)
    val dao = dout * sigmoid(ao).rsub(1) * sigmoid(ao)
    val dag = dg * pow(tanh(ag), 2).rsub(1)
    val da = Nd4j.hstack(dai, daf, dao, dag)
    val dx = da.mmul(data.w.T)
    val dprevH = da.mmul(data.wh.T)
    val db = da.sum(0)
    val dw = data.x.T.mmul(da)
    val dwh = data.prevH.T.mmul(da)
    (dx, dprevH, dprevC, dw, dwh, db)


  }


  override def toString(): String = s"number of input = ${nbInput} number of output = ${nbOutput}"

}
