package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, Identity, ReLU}
import com.cpuheater.deepdive.nn.Conv2d
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

    val config = Conv2d(height = height,
      width = width,
      channels = channels,
      nbOfFilters = nbOfFilters,
      filterHeight = filterHeight,
      filterWidth = filterWidth,
      stride = stride,
      padding = padding,
      activation = ReLU,
      name = "")

    val layerNb = 1

    val x =  Nd4j.linspace(-0.1, 0.5, 2*3*4*4).reshape(batchSize, channels, height, width)
    val w =  Nd4j.linspace(-0.2, 0.3, 3*3*4*4).reshape(nbOfFilters, channels, filterHeight, filterWidth)
    val b = Nd4j.linspace(-0.1, 0.2, nbOfFilters)
    val dout = Nd4j.linspace(-0.1, 0.2, 24).reshape(2, 3, 2, 2)


    val params =  mutable.Map[String, INDArray]()
    params(ParamType.toString(ParamType.W, layerNb)) = w
    params(ParamType.toString(ParamType.B, layerNb)) = b
    val layer  = new ConvLayer(config = config, params, 1)

    val out = layer.forward(x)

    ArrayUtil.equals(out.data().asFloat(), Array(0.0,0.0,0.0,0.0,0.21027091,0.21661097,0.22847627,0.23004639,0.5081399,0.54309976,0.64082444,0.6710144,0.0,0.0,0.0,0.0,0.6910836,0.6688039,0.5948098,0.56776005,2.3627033,2.3690434,2.3809085,2.3824787)) should be(true)

    val GradResult(dx, grads) = layer.backward(dout)

    val dw = grads(s"${ParamType.W}${layerNb}")
    val db = grads(s"${ParamType.B}${layerNb}")

    ArrayUtil.equals(dx.data().asFloat(), Array(0.0013681978,0.006166009,0.006044391,0.005010642,0.004177563,0.015214354,0.015153544,0.011553664,0.00332624,0.014971118,0.014910309,0.012161752,0.0029309825,0.010021286,0.010082094,0.0073943464,0.023040438,0.049510494,0.051577993,0.027777439,0.04752205,0.10190333,0.10622075,0.05708726,0.055427186,0.119173005,0.12349043,0.06645182,0.02898146,0.06212224,0.06437216,0.034539375,-0.0010641518,0.004220129,0.0040985118,0.005497112,7.7227224E-4,0.014241412,0.014180604,0.013986016,-7.90481E-5,0.01399818,0.013937371,0.014594104,0.0019580424,0.010994226,0.011055035,0.009340227,0.03812101,0.08259046,0.08465795,0.045776833,0.0791426,0.17098208,0.17529951,0.094545454,0.08704774,0.18825176,0.19256917,0.103910014,0.04552144,0.098121025,0.10037094,0.05399818,-0.003496502,0.0022742494,0.002152632,0.0059835818,-0.002633017,0.013268477,0.013207667,0.016418366,-0.0034843392,0.01302524,0.012964431,0.017026454,9.851023E-4,0.011967167,0.012027975,0.011286108,0.053201586,0.11567043,0.11773792,0.063776225,0.11076316,0.24006082,0.24437825,0.13200366,0.118668295,0.2573305,0.26164794,0.14136821,0.06206143,0.13411981,0.13636974,0.07345699)) should be(true)
    ArrayUtil.equals(dw.data().asFloat(), Array(0.0,0.035295203,0.043972544,0.0,0.06858582,0.08594051,0.0,0.070178494,0.08885126,0.0,0.03383982,0.043176204,0.0,0.0675698,0.084924504,0.0,0.13113046,0.16583984,0.0,0.13365677,0.17100231,0.0,0.06432952,0.083002284,0.0,0.073281474,0.09590848,0.0,0.14123571,0.18648973,0.0,0.14376204,0.1916522,0.0,0.06872311,0.09266819,0.0,0.03315332,0.044466827,0.0,0.06364302,0.086270034,0.0,0.06457667,0.088521756,0.0,0.030709386,0.04268192,0.0,0.0493547,0.06857666,0.0,0.09406866,0.1325126,0.0,0.09566134,0.13542335,0.0,0.045263164,0.065144174,0.0,0.09041649,0.12886043,0.0,0.17155151,0.24843939,0.0,0.17407782,0.25360185,0.0,0.081903905,0.121665925,0.0,0.096128166,0.13984442,0.0,0.18165678,0.26908925,0.0,0.18418309,0.27425176,0.0,0.0862975,0.13133182,0.0,0.041940507,0.063798636,0.0,0.07858125,0.122297496,0.0,0.07951489,0.12454922,0.0,0.036860418,0.05937758,0.0,0.0634142,0.09318079,0.0,0.11955151,0.17908469,0.0,0.121144176,0.18199542,0.0,0.056686502,0.08711213,0.0,0.113263175,0.17279637,0.0,0.21197256,0.33103892,0.0,0.21449888,0.3362014,0.0,0.099478275,0.16032954,0.0,0.11897485,0.18378034,0.0,0.22207783,0.35168883,0.0,0.22460414,0.35685128,0.0,0.10387187,0.16999543,0.0,0.05072769,0.08313044,0.0,0.093519464,0.15832496,0.0,0.0944531,0.16057667,0.0,0.04301145,0.07607323)) should be(true)
    ArrayUtil.equals(db.data().asFloat(), Array(0.0,0.40000007,0.81739134)) should be(true)
  }

}
