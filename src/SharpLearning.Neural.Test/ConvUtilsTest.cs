using System;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Neural.Test;

[TestClass]
public class ConvUtilsTest
{
    [TestMethod]
    public void ConvUtils_Batch_Im2Cols()
    {
        var batchSize = 5;

        var filterHeight = 2;
        var filterWidth = 2;

        var stride = 1;
        var padding = 0;

        var inputWidth = 3;
        var inputHeight = 3;
        var inputDepth = 3;

        var random = new Random(42);
        var input = Matrix<float>.Build.Random(batchSize, inputWidth * inputHeight * inputDepth, 42);

        var filterGridWidth = ConvUtils.GetFilterGridLength(inputWidth, filterWidth, stride,
            padding, BorderMode.Valid);

        var filterGridHeight = ConvUtils.GetFilterGridLength(inputHeight, filterHeight, stride,
            padding, BorderMode.Valid);

        var k = filterWidth * filterHeight * inputDepth;
        var n = batchSize * filterGridWidth * filterGridHeight;

        var actual = Matrix<float>.Build.Dense(k, n);

        ConvUtils.Batch_Im2Col(input, inputDepth, inputHeight, inputWidth, filterHeight, filterWidth,
            padding, padding, stride, stride, BorderMode.Valid, actual);

        Trace.WriteLine(actual.ToString());
        Trace.WriteLine(string.Join(",", actual.ToColumnMajorArray()));

        var expected = Matrix<float>.Build.Dense(k, n, [0.408388f, -0.5256838f, -1.416015f, -0.3205518f, 0.8964508f, -0.7706847f, 0.1228476f, 1.401819f, 0.02538049f, 0.4443011f, 0.3597376f, -0.8992839f, -0.5256838f, -0.8472909f, -0.3205518f, 0.168334f, -0.7706847f, -0.2688324f, 1.401819f, 0.5753565f, 0.4443011f, -0.8027026f, -0.8992839f, -0.6576554f, -1.416015f, -0.3205518f, 0.1622419f, -0.8718526f, 0.1228476f, 1.401819f, -0.8105127f, -1.366049f, 0.3597376f, -0.8992839f, -0.09693441f, 0.1117831f, -0.3205518f, 0.168334f, -0.8718526f, 2.464335f, 1.401819f, 0.5753565f, -1.366049f, 0.7328596f, -0.8992839f, -0.6576554f, 0.1117831f, -2.00572f, -0.8723587f, 1.785321f, 0.02021696f, -1.087396f, -0.7902505f, -0.06449615f, -0.4799407f, 0.7755837f, -0.08005979f, -0.163763f, 1.463557f, -0.5891034f, 1.785321f, -0.7747191f, -1.087396f, 1.942754f, -0.06449615f, 0.08791012f, 0.7755837f, 1.559499f, -0.163763f, 1.144407f, -0.5891034f, 1.486937f, 0.02021696f, -1.087396f, 1.386084f, -0.742821f, -0.4799407f, 0.7755837f, -0.93938f, 0.4403726f, 1.463557f, -0.5891034f, 0.2961742f, -1.676224f, -1.087396f, 1.942754f, -0.742821f, 0.3750592f, 0.7755837f, 1.559499f, 0.4403726f, 1.018316f, -0.5891034f, 1.486937f, -1.676224f, 0.5095494f, -1.069885f, 0.1028096f, -0.5383296f, -0.5273784f, -1.362978f, -2.817736f, -0.3506753f, -2.379571f, -0.205604f, -0.8553149f, 1.364009f, 1.960906f, 0.1028096f, 0.06300805f, -0.5273784f, 0.1655738f, -2.817736f, -0.2654593f, -2.379571f, 0.3019102f, -0.8553149f, 0.380102f, 1.960906f, -1.644088f, -0.5383296f, -0.5273784f, 1.407161f, 0.8093351f, -0.3506753f, -2.379571f, -0.1132597f, 0.00849107f, 1.364009f, 1.960906f, -1.907569f, 1.585406f, -0.5273784f, 0.1655738f, 0.8093351f, -0.5961999f, -2.379571f, 0.3019102f, 0.00849107f, -0.9973568f, 1.960906f, -1.644088f, 1.585406f, 0.1513373f, 0.06503697f, -0.6606446f, 1.281655f, 0.2639574f, -0.3281617f, 0.6252633f, -0.9870397f, -0.2739736f, 0.5706424f, -0.6933832f, -0.9226705f, 1.837471f, -0.6606446f, -2.021355f, 0.2639574f, -1.713513f, 0.6252633f, -0.6887951f, -0.2739736f, -0.1102718f, -0.6933832f, -0.2514778f, 1.837471f, 1.012506f, 1.281655f, 0.2639574f, -0.6539868f, -1.332823f, -0.9870397f, -0.2739736f, -0.6845301f, 0.3220822f, -0.9226705f, 1.837471f, 2.257283f, -0.2592173f, 0.2639574f, -1.713513f, -1.332823f, -0.1056926f, -0.2739736f, -0.1102718f, 0.3220822f, 0.02583288f, 1.837471f, 1.012506f, -0.2592173f, 0.5775524f, -0.734176f, 0.5288628f, 0.314957f, 1.331584f, 0.1659867f, -0.0002207408f, -0.3023876f, 0.5506561f, -1.365916f, -0.314546f, -0.6079422f, 0.3696074f, 0.5288628f, -0.7030032f, 1.331584f, 0.7429405f, -0.0002207408f, -2.21279f, 0.5506561f, 0.5057944f, -0.314546f, -1.749763f, 0.3696074f, -0.1464183f, 0.314957f, 1.331584f, 0.2864983f, 0.9384909f, -0.3023876f, 0.5506561f, 1.133461f, 1.134041f, -0.6079422f, 0.3696074f, 0.2236174f, -0.9724815f, 1.331584f, 0.7429405f, 0.9384909f, 1.441582f, 0.5506561f, 0.5057944f, 1.134041f, 0.2430595f, 0.3696074f, -0.1464183f, -0.9724815f, 0.7229092f]);

        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void ConvUtils_Batch_Col2Im()
    {
        var batchSize = 5;

        var filterHeight = 2;
        var filterWidth = 2;

        var stride = 1;
        var padding = 0;

        var inputWidth = 3;
        var inputHeight = 3;
        var inputDepth = 3;

        var filterGridWidth = ConvUtils.GetFilterGridLength(inputWidth, filterWidth, stride,
            padding, BorderMode.Valid);

        var filterGridHeight = ConvUtils.GetFilterGridLength(inputHeight, filterHeight, stride,
            padding, BorderMode.Valid);

        var k = filterWidth * filterHeight * inputDepth;
        var n = filterGridWidth * filterGridHeight * batchSize;
        var fanIn = inputWidth * inputHeight * inputDepth;

        var input = Matrix<float>.Build.Random(k, n, 42);
        var actual = Matrix<float>.Build.Dense(batchSize, fanIn);

        ConvUtils.Batch_Col2Im(input, inputDepth, inputHeight, inputWidth,
            filterHeight, filterWidth, padding, padding, stride, stride, BorderMode.Valid, actual);

        Trace.WriteLine(actual.ToString());
        Trace.WriteLine(string.Join(",", actual.ToColumnMajorArray()));

        var expected = Matrix<float>.Build.Dense(batchSize, fanIn, [0.408388f, -0.3281617f, -0.163763f, -0.7540793f, -0.8690567f, -0.8093507f, 0.2888344f, -1.777985f, -2.136633f, 2.92046f, -2.021355f, -0.4799407f, -0.6079422f, 0.5664175f, 1.640147f, 0.2616988f, -0.4687745f, -0.7903177f, 1.407904f, 0.1495381f, -1.212453f, 0.6085976f, -0.7663184f, -0.05670342f, 1.895431f, -0.6066797f, -0.2541801f, -0.01155096f, 1.438064f, -1.349128f, 1.942754f, 0.5057944f, -1.907569f, -0.5227588f, 0.5727027f, -1.167249f, 0.2078037f, 2.980192f, 0.4892522f, -0.6720377f, 0.9384909f, -0.9973568f, 0.5546624f, 1.710745f, 1.995577f, -0.734176f, -2.817736f, -0.8027026f, -0.7883626f, -1.275902f, -0.5054669f, 0.3228757f, 3.105314f, -0.3089013f, 1.549119f, -0.5383296f, 1.401819f, 1.837471f, 0.1251182f, -0.7002729f, 0.07180786f, -0.9396007f, 0.6037194f, -0.7305622f, 1.063156f, 4.591741f, 0.4193244f, -1.031005f, -3.045349f, 0.4254266f, 0.6900162f, -2.136511f, -1.578628f, 0.7839373f, 1.781849f, 0.1622419f, -0.6845301f, -1.676224f, 1.028266f, 0.9345228f, 0.789884f, 1.158841f, 1.703116f, -0.8997472f, -1.423375f, -0.1056926f, -0.08005979f, 1.399474f, -0.05612089f, -0.722365f, -0.6606446f, 0.08791012f, -1.749763f, 0.685056f, 0.3641174f, 0.2083111f, -0.5394329f, 1.846675f, 0.5931945f, -1.26804f, -1.087396f, 0.5506561f, -1.644088f, -0.8753259f, -1.839462f, 0.5598704f, -2.054844f, 1.20434f, -3.263947f, 1.221963f, -0.5145022f, -1.402665f, 1.101824f, 0.4248552f, -2.63849f, 1.160408f, 2.130142f, 0.3172536f, 1.109406f, 0.9979748f, 0.2864983f, 0.00849107f, -2.00572f, 1.178588f, -0.3127078f, -1.662103f, -1.043834f, 1.065703f, -0.9702578f, -0.1781971f, -1.362978f, 0.4443011f, -1.050083f, 0.6755545f, -1.088875f]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void ConvUtils_ReshapeConvolutionsToRowMajor()
    {
        var batchSize = 5;

        var filterHeight = 2;
        var filterWidth = 2;
        var filterDepth = 2;

        var stride = 1;
        var padding = 0;

        var inputWidth = 3;
        var inputHeight = 3;
        var inputDepth = 3;

        var filterGridWidth = ConvUtils.GetFilterGridLength(inputWidth, filterWidth, stride,
            padding, BorderMode.Valid);

        var filterGridHeight = ConvUtils.GetFilterGridLength(inputHeight, filterHeight, stride,
            padding, BorderMode.Valid);

        var k = filterDepth;
        var crs = inputDepth * filterWidth * filterHeight;
        var npq = batchSize * filterGridWidth * filterGridHeight;

        var convolutedInput = Matrix<float>.Build.Dense(k, npq, [-6.260461f, 87.38299f, -7.173417f, 94.47046f, -8.999331f, 108.6454f, -9.912288f, 115.7329f, -6.260461f, 87.38299f, -7.173417f, 94.47046f, -8.999331f, 108.6454f, -9.912288f, 115.7329f, -6.260461f, 87.38299f, -7.173417f, 94.47046f, -8.999331f, 108.6454f, -9.912288f, 115.7329f, -6.260461f, 87.38299f, -7.173417f, 94.47046f, -8.999331f, 108.6454f, -9.912288f, 115.7329f, -6.260461f, 87.38299f, -7.173417f, 94.47046f, -8.999331f, 108.6454f, -9.912288f, 115.7329f]);
        var actual = Matrix<float>.Build.Dense(batchSize, k * filterGridWidth * filterGridHeight);

        ConvUtils.ReshapeConvolutionsToRowMajor(convolutedInput, inputDepth, inputHeight, inputWidth,
            filterHeight, filterWidth, padding, padding, stride, stride, BorderMode.Valid, actual);

        var expected = Matrix<float>.Build.Dense(batchSize, k * filterGridWidth * filterGridHeight, [-6.260461f, -6.260461f, -6.260461f, -6.260461f, -6.260461f, -7.173417f, -7.173417f, -7.173417f, -7.173417f, -7.173417f, -8.999331f, -8.999331f, -8.999331f, -8.999331f, -8.999331f, -9.912288f, -9.912288f, -9.912288f, -9.912288f, -9.912288f, 87.38299f, 87.38299f, 87.38299f, 87.38299f, 87.38299f, 94.47046f, 94.47046f, 94.47046f, 94.47046f, 94.47046f, 108.6454f, 108.6454f, 108.6454f, 108.6454f, 108.6454f, 115.7329f, 115.7329f, 115.7329f, 115.7329f, 115.7329f]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void ConvUtils_ReshapeRowMajorToConvolutionLayout()
    {
        var batchSize = 5;

        var filterHeight = 2;
        var filterWidth = 2;
        var filterDepth = 2;

        var stride = 1;
        var padding = 0;

        var inputWidth = 3;
        var inputHeight = 3;
        var inputDepth = 3;

        var filterGridWidth = ConvUtils.GetFilterGridLength(inputWidth, filterWidth, stride,
            padding, BorderMode.Valid);

        var filterGridHeight = ConvUtils.GetFilterGridLength(inputHeight, filterHeight, stride,
            padding, BorderMode.Valid);

        var k = filterDepth;
        var crs = inputDepth * filterWidth * filterHeight;
        var npq = batchSize * filterGridWidth * filterGridHeight;

        var rowMajor = Matrix<float>.Build.Dense(batchSize, k * filterGridWidth * filterGridHeight, [-6.260461f, -6.260461f, -6.260461f, -6.260461f, -6.260461f, -7.173417f, -7.173417f, -7.173417f, -7.173417f, -7.173417f, -8.999331f, -8.999331f, -8.999331f, -8.999331f, -8.999331f, -9.912288f, -9.912288f, -9.912288f, -9.912288f, -9.912288f, 87.38299f, 87.38299f, 87.38299f, 87.38299f, 87.38299f, 94.47046f, 94.47046f, 94.47046f, 94.47046f, 94.47046f, 108.6454f, 108.6454f, 108.6454f, 108.6454f, 108.6454f, 115.7329f, 115.7329f, 115.7329f, 115.7329f, 115.7329f]);
        var actual = Matrix<float>.Build.Dense(k, npq);

        ConvUtils.ReshapeRowMajorToConvolutionLayout(rowMajor, inputDepth, inputHeight, inputWidth,
            filterHeight, filterWidth, padding, padding, stride, stride, BorderMode.Valid, actual);

        var expected = Matrix<float>.Build.Dense(k, npq, [-6.260461f, 87.38299f, -7.173417f, 94.47046f, -8.999331f, 108.6454f, -9.912288f, 115.7329f, -6.260461f, 87.38299f, -7.173417f, 94.47046f, -8.999331f, 108.6454f, -9.912288f, 115.7329f, -6.260461f, 87.38299f, -7.173417f, 94.47046f, -8.999331f, 108.6454f, -9.912288f, 115.7329f, -6.260461f, 87.38299f, -7.173417f, 94.47046f, -8.999331f, 108.6454f, -9.912288f, 115.7329f, -6.260461f, 87.38299f, -7.173417f, 94.47046f, -8.999331f, 108.6454f, -9.912288f, 115.7329f]);

        MatrixAsserts.AreEqual(expected, actual);
    }
}
