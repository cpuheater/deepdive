package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, ReLU}
import com.cpuheater.deepdive.nn.{Linear, RNN}
import com.cpuheater.deepdive.nn.layers.ParamType.PreOutput
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

class RNNLayer(layerConfig: RNN,
                  override val params: mutable.Map[String, INDArray],
                  layerNb: Int) extends Layer {

  private val cache: mutable.Map[String, INDArray] = mutable.Map[String, INDArray]()
  private val cacheTS: mutable.ListBuffer[Map[String, INDArray]] = new ListBuffer[Map[String, INDArray]]()

  private case class FFData(x: INDArray, out: INDArray, preOutput: INDArray, prevHidden: INDArray, w: INDArray, wh: INDArray)

  override def name: String = layerConfig.name

  override def activationFn: ActivationFn = layerConfig.activation

  def nbOutput: Int = layerConfig.nbOutput

  def nbInput: Int = layerConfig.nbInput

  override def backward(dout: INDArray, isTraining: Boolean = true): GradResult = {
     ???
  }


  override def forward(x: INDArray, isTraining: Boolean =  true): INDArray = {
    val (out, _)  = innerForward(x, isTraining)
    out
  }

  private def innerForward(x: INDArray, isTraining: Boolean =  true): (INDArray, List[FFData]) = {
    val w = params(ParamType.toString(ParamType.W, layerNb))
    val b = params(ParamType.toString(ParamType.B, layerNb))
    val wh = params(ParamType.toString(ParamType.WH, layerNb))
    val prevHidden = params(ParamType.toString(ParamType.H, layerNb))

    val Array(n, t, d) = x.shape()
    val h = wh.shape()(0)

    val allOut = Nd4j.zeros(Array(n, t, h):_*)

    val allData = ListBuffer[FFData]()

    (0 until t).foreach{
      time =>
        val currentX = x.tensorAlongDimension(time, 0, 2)
        if(time == 0){
          val data = forwardTimeStep(currentX, prevHidden, w, wh, b)
          allData += data
          allOut.tensorAlongDimension(time, 0, 2).assign(data.out)
        } else {
          val prevHidden = allOut.tensorAlongDimension(time-1, 0, 2)
          val data = forwardTimeStep(currentX, prevHidden, w, wh, b)
          allData +=data
          allOut.tensorAlongDimension(time, 0, 2).assign(data.out)
        }
    }

    (allOut, allData.toList)
  }

  private def forwardTimeStep(x: INDArray,
                          prevHidden: INDArray,
                          w: INDArray,
                          wh: INDArray,
                          b: INDArray): FFData = {

    val preOutput = (x.dot(w) + prevHidden.dot(wh)).addRowVector(b)
    val out = activationFn(preOutput)
    //x: INDArray, out: INDArray, preOutput: INDArray, prevHidden: INDArray, w: INDArray, wh: INDArray

    FFData(x, out, preOutput, prevHidden, w, wh)
  }

  def backwardNew(dout: INDArray, x: INDArray, isTraining: Boolean = true): GradResult = {

    val (_, allData) = innerForward(x, isTraining)

    val w = params(ParamType.toString(ParamType.W, layerNb))
    val b = params(ParamType.toString(ParamType.B, layerNb))
    val wh = params(ParamType.toString(ParamType.WH, layerNb))
    var prevHidden = params(ParamType.toString(ParamType.H, layerNb))

    val Array(n, t, d) = dout.shape()
    val h = wh.shape()(0)

    var dx = Nd4j.zeros(Array(n, t, d): _*)
    var dw =  Nd4j.zeros(d, h)
    var dwh = Nd4j.zeros(h, h)
    var db = Nd4j.zeros(h)

    (t-1 to 0).by(-1).foreach {
      time =>
        val ula = dout.tensorAlongDimension(time, 0, 2)
        val doutTS = dout.tensorAlongDimension(time, 0, 2) + prevHidden

        val (dxTS, dhiddenTS, dwTS, dwhTS, dbTS) = backwardTimeStep(doutTS, allData(time))
        dx.tensorAlongDimension(time, 0, 2).assign(dxTS)
        dw = dw + dwTS
        dwh = dwh + dwhTS
        db = db + dbTS
        prevHidden = dhiddenTS
    }

    val grads = Map(s"${ParamType.W}${layerNb}" ->dw,
      s"${ParamType.B}${layerNb}"->db,
      s"${ParamType.WH}${layerNb}"->dwh)
    GradResult(dx, grads)

  }


  private def backwardTimeStep(dout: INDArray,
                               cache: FFData,
                               /*preOutput: INDArray,
                               prevHidden: INDArray,
                               w: INDArray,
                               wh: INDArray,
                               b: INDArray,*/
                               isTraining: Boolean = true): (INDArray, INDArray, INDArray, INDArray, INDArray) = {


    val da = activationFn.derivative(cache.preOutput.dup()) * dout

    val dx = da.dot(cache.w.T)
    val dhidden = da.dot(cache.wh.T)
    val dw = cache.x.T.dot(da)
    val dwh = cache.prevHidden.T.dot(da)
    val db = Nd4j.sum(da, 0)
    (dx, dhidden, dw, dwh, db)
  }

  override def toString(): String = s"number of input = ${nbInput} number of output = ${nbOutput}"

}
