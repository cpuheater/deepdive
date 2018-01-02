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
    params(ParamType.print(ParamType.W, layerNb)) = w
    params(ParamType.print(ParamType.B, layerNb)) = b
    val layer  = new ConvLayer(config = config, params, 1)

    val out = layer.forward(x)

    ArrayUtil.equals(out.data().asFloat(), Array(-0.087598115, -0.9805359, -0.109877825, -1.0314355, -0.18387194, -1.191289, -0.21092162, -1.2469586, 0.21027091, 0.6910836, 0.21661097, 0.6688039, 0.22847627, 0.5948098, 0.23004639, 0.56776005, 0.5081399, 2.3627033, 0.54309976, 2.3690434, 0.64082444, 2.3809085, 0.6710144, 2.3824787)) should be(true)

    val (dx, dw, db) = layer.backward(dout)

    ArrayUtil.equals(dx.data().asFloat(), Array(0.019619947,0.03781697,0.03708726,0.01824263,0.038117975,0.07339009,0.07220432,0.035500154,0.034651868,0.06864701,0.06746124,0.03422317,0.017585894,0.034843422,0.034387358,0.017439954,0.012724234,0.024025539,0.026579507,0.012988746,0.024326548,0.045807242,0.05118881,0.024992393,0.03399514,0.067333534,0.07271512,0.036850102,0.01725753,0.03418669,0.037014294,0.01875342,0.01159319,0.026141688,0.025411982,0.014594102,0.024253573,0.05441776,0.05323199,0.030392218,0.020787477,0.049674682,0.048488908,0.029115237,0.0117482515,0.027546369,0.027090304,0.015980544,0.030966863,0.06488903,0.067443,0.035609607,0.06300092,0.13191244,0.13729404,0.07242323,0.072669506,0.15343875,0.1588203,0.08428094,0.037689276,0.079428405,0.08225602,0.043563396,0.0035664355,0.014466406,0.0137367025,0.010945577,0.01038918,0.035445433,0.034259662,0.025284285,0.00692308,0.030702349,0.029516574,0.024007302,0.0059106126,0.02024932,0.019793253,0.014521134,0.04920949,0.10575251,0.10830648,0.058230467,0.10167529,0.21801764,0.22339924,0.11985406,0.111343876,0.23954396,0.2449255,0.13171178,0.058121018,0.124670126,0.12749773,0.06837337)) should be(true)
    ArrayUtil.equals(dw.data().asFloat(), Array(0.026617855,0.035295203,0.043972544,0.05123113,0.06858582,0.08594051,0.05150573,0.070178494,0.08885126,0.02450344,0.03383982,0.043176204,0.05021511,0.0675698,0.084924504,0.09642107,0.13113046,0.16583984,0.096311234,0.13365677,0.17100231,0.045656756,0.06432952,0.083002284,0.05065447,0.073281474,0.09590848,0.09598171,0.14123571,0.18648973,0.09587188,0.14376204,0.1916522,0.044778034,0.06872311,0.09266819,0.02183982,0.03315332,0.044466827,0.04101602,0.06364302,0.086270034,0.040631585,0.06457667,0.088521756,0.018736841,0.030709386,0.04268192,0.030132728,0.0493547,0.06857666,0.05562473,0.09406866,0.1325126,0.05589933,0.09566134,0.13542335,0.02538216,0.045263164,0.065144174,0.05197255,0.09041649,0.12886043,0.094663635,0.17155151,0.24843939,0.0945538,0.17407782,0.25360185,0.042141885,0.081903905,0.121665925,0.05241191,0.096128166,0.13984442,0.094224274,0.18165678,0.26908925,0.09411444,0.18418309,0.27425176,0.041263167,0.0862975,0.13133182,0.020082382,0.041940507,0.063798636,0.034864992,0.07858125,0.122297496,0.03448055,0.07951489,0.12454922,0.014343249,0.036860418,0.05937758,0.033647608,0.0634142,0.09318079,0.060018327,0.11955151,0.17908469,0.060292926,0.121144176,0.18199542,0.026260879,0.056686502,0.08711213,0.053729985,0.113263175,0.17279637,0.0929062,0.21197256,0.33103892,0.092796355,0.21449888,0.3362014,0.03862701,0.099478275,0.16032954,0.05416935,0.11897485,0.18378034,0.092466846,0.22207783,0.35168883,0.092356995,0.22460414,0.35685128,0.037748292,0.10387187,0.16999543,0.018324945,0.05072769,0.08313044,0.02871396,0.093519464,0.15832496,0.028329518,0.0944531,0.16057667,0.009949654,0.04301145,0.07607323)) should be(true)
    ArrayUtil.equals(db.data().asFloat(), Array(-0.017391264,0.40000007,0.81739134)) should be(true)
  }


}
