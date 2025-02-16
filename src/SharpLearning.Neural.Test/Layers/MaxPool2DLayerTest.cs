using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Layers;

namespace SharpLearning.Neural.Test.Layers;

[TestClass]
public class MaxPool2DLayerTest
{
    [TestMethod]
    public void MaxPool2DLayer_CopyLayerForPredictionModel()
    {
        var batchSize = 1;

        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height, 1, 5, 6);
        sut.Initialize(3, 3, 1, batchSize, Initialization.GlorotUniform, new Random(232));

        var layers = new List<ILayer>();
        sut.CopyLayerForPredictionModel(layers);

        var actual = (MaxPool2DLayer)layers.Single();

        Assert.AreEqual(sut.Width, actual.Width);
        Assert.AreEqual(sut.Height, actual.Height);
        Assert.AreEqual(sut.Depth, actual.Depth);
        Assert.AreEqual(sut.BorderMode, actual.BorderMode);

        Assert.AreEqual(sut.InputWidth, actual.InputWidth);
        Assert.AreEqual(sut.InputHeight, actual.InputHeight);
        Assert.AreEqual(sut.InputDepth, actual.InputDepth);

        Assert.AreEqual(sut.Switchx.Length, actual.Switchx.Length);
        Assert.AreEqual(sut.Switchx[0].Length, actual.Switchx[0].Length);

        Assert.AreEqual(sut.Switchy.Length, actual.Switchy.Length);
        Assert.AreEqual(sut.Switchy[0].Length, actual.Switchy[0].Length);

        Assert.AreEqual(sut.OutputActivations.RowCount, actual.OutputActivations.RowCount);
        Assert.AreEqual(sut.OutputActivations.ColumnCount, actual.OutputActivations.ColumnCount);
    }

    [TestMethod]
    public void MaxPool2DLayer_Forward()
    {
        const int inputWidth = 20;
        const int inputHeight = 20;
        const int inputDepth = 2;
        const int fanIn = inputWidth * inputHeight * inputDepth;
        const int fanOut = 10 * 10 * 2;

        const int batchSize = 1;
        var random = new Random(32);

        // Create layer
        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height);
        sut.Initialize(inputWidth, inputHeight, inputDepth, batchSize,
            Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        var actual = sut.Forward(input);

        Trace.WriteLine(actual.ToString());
        Trace.WriteLine(string.Join(",", actual.ToColumnMajorArray()));

        var expected = Matrix<float>.Build.Dense(batchSize, fanOut, [1.737387f, 0.8429711f, 1.403809f, 0.7437097f, 2.154061f, 1.737234f, 0.9562361f, 0.1203717f, 1.253413f, 0.7093143f, 1.521145f, 1.494283f, 1.988454f, 2.392222f, 1.740699f, 0.7398372f, 0.9477737f, 1.73438f, 1.367192f, 0.7603366f, 1.583967f, 0.3866753f, 1.930486f, 1.501988f, -0.2519213f, 0.9616809f, 0.8507621f, 0.5222243f, 0.528969f, 1.083474f, 0.5110471f, 1.111015f, 0.4116569f, 1.012139f, 1.541237f, 1.286736f, 0.8889436f, 0.6083445f, 1.407371f, 1.033507f, 0.2372739f, 1.175704f, 0.3457479f, 0.3563888f, 0.4308063f, 2.15408f, 1.019059f, 1.69062f, 0.5580661f, 0.9991792f, 0.8225476f, 0.1575469f, 1.119048f, -0.03910597f, 1.736111f, 0.7009985f, 0.1849347f, 1.268318f, 1.533113f, 0.891203f, 0.7703907f, 0.7964001f, 2.104593f, 3.125018f, 0.4306072f, 1.297616f, 0.8612604f, 1.569523f, 1.496838f, 0.7015814f, 0.7657425f, 0.8277726f, 0.3020416f, 1.502974f, 0.9276986f, 0.9929756f, 0.9644111f, 0.7933079f, 0.9039445f, 0.4037939f, 0.6111527f, -0.02752473f, 0.8821087f, 1.149586f, 0.2484843f, 0.8898949f, 1.909704f, 1.046652f, 1.395888f, 1.341396f, 3.130147f, 1.424874f, -0.1669227f, 1.688097f, 1.319619f, 0.08981451f, 1.955076f, 1.188523f, 0.9187648f, 1.701037f, 1.126729f, 0.6088547f, 1.249962f, 1.904854f, 1.216359f, 0.8841643f, 0.9773607f, 0.5250032f, 2.041504f, 1.75729f, 0.2925639f, 1.233287f, 0.6095849f, 0.9424562f, 1.445586f, 0.5931875f, 1.458192f, 0.4289872f, 0.5092088f, 1.496163f, 1.205378f, 1.003089f, -0.5055257f, 0.9426916f, 1.97264f, 1.179588f, 1.628175f, 2.082574f, 0.478283f, 0.6607473f, 1.860639f, 1.452383f, 2.17662f, 1.086674f, 2.466586f, 0.1421053f, 1.238979f, 0.8957624f, 0.6944376f, 1.249352f, 0.7237418f, 3.043795f, 1.631333f, 0.7378432f, 0.6678889f, 1.090085f, 1.857423f, 1.000153f, 1.650252f, 1.500757f, 2.024655f, 0.9628305f, 0.8909977f, -0.7175303f, 2.396366f, 1.028608f, 0.7338257f, 0.9764791f, 0.5674692f, 1.814738f, 0.7745261f, 0.5802411f, 0.142071f, 0.9685609f, 0.05501625f, 1.262817f, 0.9647988f, 1.111344f, -0.2743198f, 1.031065f, 0.8540451f, 0.633197f, 0.8172408f, -0.6463516f, 0.6572174f, 0.5348259f, 0.4829673f, 0.7212811f, 0.9138665f, 1.560033f, 1.193395f, 0.6193073f, 0.4542928f, 2.111476f, 0.7224295f, 0.2179742f, 0.3198487f, 1.163711f, 1.428939f, 1.220046f, 0.1001558f, 0.7708471f, 1.356724f, 0.3361169f, -0.3378747f, 1.28403f, 0.6157113f, 1.262698f, 1.797522f, 1.135491f]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void MaxPool2DLayer_Forward_2()
    {
        const int inputWidth = 4;
        const int inputHeight = 4;
        const int inputDepth = 1;
        const int fanIn = inputWidth * inputHeight * inputDepth;
        const int fanOut = 2 * 2 * 1;

        const int batchSize = 1;

        // Create layer
        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height);
        sut.Initialize(inputWidth, inputHeight, inputDepth, batchSize,
            Initialization.GlorotUniform, new Random(232));

        var inputData = new float[] { 3, 0, 0, 6, 0, 2, 3, 0, 0, 8, 10, 0, 4, 6, 0, 7 };
        var input = Matrix<float>.Build.Dense(batchSize, fanIn, inputData);
        Trace.WriteLine(input.ToString());
        var actual = sut.Forward(input);

        var expected = Matrix<float>.Build.Dense(batchSize, fanOut, [3, 6, 8, 10]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void MaxPool2DLayer_Forward_3()
    {
        const int inputWidth = 4;
        const int inputHeight = 4;
        const int inputDepth = 2;
        const int fanIn = inputWidth * inputHeight * inputDepth;
        const int fanOut = 2 * 2 * 2;

        const int batchSize = 1;

        // Create layer
        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height);
        sut.Initialize(inputWidth, inputHeight, inputDepth, batchSize,
            Initialization.GlorotUniform, new Random(232));

        var inputData = new float[] { 3, 0, 0, 6, 0, 2, 3, 0, 0, 8, 10, 0, 4, 6, 0, 7, 4, 0, 2, 0, 0, 8, 3, 5, 10, 0, 12, 0, 6, 5, 3, 2 };
        var input = Matrix<float>.Build.Dense(batchSize, fanIn, inputData);
        Trace.WriteLine(input.ToString());
        var actual = sut.Forward(input);

        var expected = Matrix<float>.Build.Dense(batchSize, fanOut, [3, 6, 8, 10, 8, 5, 10, 12]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void MaxPool2DLayer_Backward()
    {
        const int inputWidth = 20;
        const int inputHeight = 20;
        const int inputDepth = 2;
        const int fanIn = inputWidth * inputHeight * inputDepth;
        const int fanOut = 10 * 10 * 2;

        const int batchSize = 1;
        var random = new Random(32);

        // Create layer
        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height);
        sut.Initialize(inputWidth, inputHeight, inputDepth, batchSize,
            Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        sut.Forward(input);

        var delta = Matrix<float>.Build.Random(batchSize, fanOut, random.Next());
        var actual = sut.Backward(delta);

        Trace.WriteLine(actual.ToString());
        Trace.WriteLine(string.Join(",", actual.ToColumnMajorArray()));

        var expected = Matrix<float>.Build.Dense(batchSize, fanIn, [0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0.8085009f, 0f, 0f, 0f, -0.01437248f, 0f, 0f, 1.041877f, 0f, 0f, 0f, 0f, 0f, 0.4979768f, 0f, -0.5938089f, 0f, -0.9181094f, 0f, -1.900491f, 0f, 0f, 0f, -0.9150107f, 0f, 0f, 0f, 0f, 0f, -0.4453017f, -0.4299661f, 0f, 0f, 0f, 0f, 0.9363991f, 0f, 0f, -0.4949541f, 0f, 0f, 0f, 0.1292399f, 0f, 0f, 0f, -0.9616904f, 0f, 0f, 0f, 0f, 0f, -1.287248f, 0f, 0f, 0f, 0.2155272f, 0f, 0f, 0f, -1.007965f, 0f, 0f, 0f, 0f, 1.076965f, 0f, 0f, -1.401237f, 0f, -1.244568f, 0f, 0f, 0f, 0f, -0.478288f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -0.6055867f, -0.1095726f, 0f, 0.3003371f, 0f, 0f, 0f, 0f, 0f, -0.3044865f, 0f, 0f, 0f, 0f, 0.8818393f, 0f, -0.4136987f, 0.4168211f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -0.1715285f, 0f, 0.3923124f, 0.2809646f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -0.5905057f, 0f, -0.09473711f, 0f, 0.3884106f, 0f, 0f, -0.4212746f, 0f, 0f, 0f, 0f, -0.9300717f, 0f, -1.464727f, -0.1085227f, 0f, 0f, 1.515902f, 0f, 0f, 0f, 0f, 0f, 0f, 1.3771f, 0f, 0f, 0f, 0f, 0f, 0f, 0.1722498f, 0f, 0f, 0.7326968f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -0.256293f, 0f, 0.005195214f, 0f, 0f, -1.809731f, 0f, 0f, 0f, 0.5915539f, 0f, 0f, -0.4030921f, 0f, -0.8363255f, -0.2891991f, 0f, -0.9076035f, 0f, 0f, 0f, 0f, 1.067826f, 1.14113f, 0f, 0f, -1.372615f, 0f, 0f, 0.02314425f, 0f, 0f, 0f, 0f, -0.5714836f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -1.125379f, 0f, 0f, 0f, 0f, 1.118231f, 0f, 0f, 0.2472252f, 0f, -0.7428527f, 0f, -1.040836f, 0f, 0f, 0f, 0.06274998f, 0f, 0.6431293f, 0f, -0.3932301f, 0f, 0f, -1.111612f, 0f, 0.7901574f, 0f, 1.980336f, 0f, 0f, 0f, 0f, 0.5354787f, 0f, 0f, 0.7546993f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -2.181903f, 0f, 0.08855283f, 0f, 0f, 0f, 0f, 0f, 1.346654f, 0f, 0f, 0.4436988f, 0f, -2.124305f, 0f, 0f, 0f, 0f, 1.103836f, 0f, -0.2725285f, 0.1360921f, 0f, 1.000088f, 0f, 0.932502f, 0f, 0f, 0f, 0.1138889f, 0f, 0f, 0f, 0f, 0f, -0.1304505f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -0.4008962f, 0f, 0f, 0f, -1.161332f, 0f, 0f, 0.3786051f, 0.2474999f, 0f, 0f, 0f, 0f, 0.885915f, 0f, 0f, -0.2077033f, 0f, 0f, 0f, 0f, 0f, -1.774759f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -0.3795716f, 0f, 0f, -1.311509f, 0f, 0f, 0f, 0f, -1.585828f, 0.5753101f, 0f, 0f, 0f, 0f, 0f, -0.978768f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -0.1397521f, 0f, 0f, 0f, 2.956711f, 0f, 0f, 0f, 0f, -1.591264f, 0f, 0.5886439f, 0f, 0f, -1.348895f, 0f, 0f, 2.115214f, 0f, -0.2732723f, 0f, 0f, 0f, -0.3678305f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -1.048669f, 1.061424f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -1.046116f, 0f, 0f, -0.4922836f, 0f, -1.362494f, 0f, 1.456946f, -0.2943774f, 0f, 0f, 0f, 0f, 0f, -0.6920757f, 0f, 0f, -0.8034356f, 0.8028942f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0.7852993f, 0f, 0f, 0f, 0.4411084f, 0f, 0.438746f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0.03683103f, 0f, 0f, 0.671003f, 0f, 0f, 0f, 0.6490055f, 0f, 0f, 0f, 0f, -0.4582749f, 0f, 0f, 0.1131398f, -1.270652f, 0f, -2.803502f, 0f, 0f, 0f, 0f, 0.4446304f, 0f, 0.3837125f, -0.6822397f, 0f, 0f, 0f, 0.090445f, 0f, -2.116256f, 0f, 0f, -1.008349f, 0f, 0f, 0f, 0f, -1.282366f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -0.3974761f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -0.6379933f, 0f, 0.3958001f, 1.004088f, 0f, 0f, 0f, 0f, 0f, 0f, -3.557195f, 0f, 0f, 0f, 0f, 0f, 0f, -1.416259f, 0f, 0.8337035f, 0f, 0f, 0f, 0f, 0f, 1.234198f, 0f, 0f, 1.57467f, 0f, 0f, -1.000447f, 0f, -0.2661186f, 0f, 1.048688f, 0f, 0f, 0f, 0f, 0f, 0f, 1.26955f, 0f, 0f, 0f, 0f, -1.462413f, 0f, 0f, 0.9360681f, 0.6391365f, 0f, 0f, 0f, 0f, 0f, 0f, -0.03548741f, 0f, 0.1278973f, 0f, -0.4136214f, 0f, 0.9968569f, -0.07145377f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1.252816f, 0f, -0.9959089f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -2.005155f, 0f, 0f, -0.8804028f, 0f, 0f, 0f, 0f, 0f, 1.159981f, 0f, 0f, 0f, 0.8770596f, 0f, 0f, 0.3886716f, 0.5398855f, 0f, 0f, 0f, 1.165788f, 0f, 0f, 0f, 0f, -0.4803754f, -0.02129833f, 0f, 0f, 0f, 1.804181f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0.3445181f, 0f, 0.702647f, 0f, 0.9873983f, 0f, 2.234645f, 0f, 0f, -0.9068208f, 0f, 0f, 0f, 0f, -0.5695083f, 0f, 0f, -0.1133842f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 2.310154f, 0f, 0f, -0.01837711f, 0f, 0f, 0f, 0f, -1.367691f, 0f, 0f, 0f, 2.204792f, 0f, 0f, -0.168677f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1.706425f, -0.627474f, 0f, 0f, 0.01406403f, 0f, 0f, -0.9384971f, 0f, 0f, 0f, 0f, -0.7298944f, 0f, -0.03289218f, 0f, -0.7163599f, 0.9871746f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0.9926041f, 0f, 0f, 0f, 0f, 1.05477f, 0f, -0.1432948f, 0f, 0f, 0f, 0f, -0.8373516f, 0f, 0f, 0f, -0.02648427f, 0f, 0f, 0f, 0f, 1.125633f, 0f, 0f, -0.1470989f, 0f, 0f, 0f, 0f, 0.7238355f, 0f, 0f, 0f, -1.240024f, 0f, 0f, 0f, 0f, 1.452529f, -0.2726488f, 0f, 0f, -0.5126494f, 0f, -0.6268897f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -0.8481783f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -1.281046f, 0f, 0f, -1.109109f, 0.5653794f, 0f, 0f, -0.7675006f, 0f, -0.6390485f, 0f, 0f, -1.11143f, 0f]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void MaxPool2DLayer_Backward_2()
    {
        const int inputWidth = 4;
        const int inputHeight = 4;
        const int inputDepth = 1;
        const int fanIn = inputWidth * inputHeight * inputDepth;
        const int fanOut = 2 * 2 * 1;

        const int batchSize = 1;
        var random = new Random(32);

        // Create layer
        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height);
        sut.Initialize(inputWidth, inputHeight, inputDepth, batchSize,
            Initialization.GlorotUniform, random);

        var inputData = new float[] { 3, 0, 0, 6, 0, 2, 3, 0, 0, 8, 10, 0, 4, 6, 0, 7 };
        var input = Matrix<float>.Build.Dense(batchSize, fanIn, inputData);
        Trace.WriteLine(input.ToString());
        sut.Forward(input);

        var delta = Matrix<float>.Build.Dense(batchSize, fanOut, 1);
        var actual = sut.Backward(delta);

        var expected = Matrix<float>.Build.Dense(batchSize, fanIn, [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void MaxPool2DLayer_Backward_3()
    {
        const int inputWidth = 4;
        const int inputHeight = 4;
        const int inputDepth = 2;
        const int fanIn = inputWidth * inputHeight * inputDepth;
        const int fanOut = 2 * 2 * 2;

        const int batchSize = 1;
        var random = new Random(32);

        // Create layer
        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height);
        sut.Initialize(inputWidth, inputHeight, inputDepth, batchSize,
            Initialization.GlorotUniform, random);

        var inputData = new float[] { 3, 0, 0, 6, 0, 2, 3, 0, 0, 8, 10, 0, 4, 6, 0, 7, 4, 0, 2, 0, 0, 8, 3, 5, 10, 0, 12, 0, 6, 5, 3, 2 };
        var input = Matrix<float>.Build.Dense(batchSize, fanIn, inputData);
        Trace.WriteLine(input.ToString());
        sut.Forward(input);

        var delta = Matrix<float>.Build.Dense(batchSize, fanOut, 1);
        var actual = sut.Backward(delta);

        var expected = Matrix<float>.Build.Dense(batchSize, fanIn, [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void MaxPool2DLayer_MultipleForwardsPasses()
    {
        const int inputWidth = 20;
        const int inputHeight = 20;
        const int inputDepth = 2;
        const int fanIn = inputWidth * inputHeight * inputDepth;
        const int fanOut = 10 * 10 * 2;

        const int batchSize = 1;
        var random = new Random(32);

        // Create layer
        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height);
        sut.Initialize(inputWidth, inputHeight, inputDepth, batchSize,
            Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        var expected = Matrix<float>.Build.Dense(batchSize, fanOut);
        sut.Forward(input).CopyTo(expected);

        for (var i = 0; i < 20; i++)
        {
            var actual = sut.Forward(input);

            Assert.AreEqual(expected, actual);
        }
    }

    [TestMethod]
    public void MaxPool2DLayer_MultipleBackwardsPasses()
    {
        const int inputWidth = 20;
        const int inputHeight = 20;
        const int inputDepth = 2;
        const int fanIn = inputWidth * inputHeight * inputDepth;
        const int fanOut = 10 * 10 * 2;

        const int batchSize = 1;
        var random = new Random(32);

        // Create layer
        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height);
        sut.Initialize(inputWidth, inputHeight, inputDepth, batchSize,
            Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        sut.Forward(input);

        var delta = Matrix<float>.Build.Dense(batchSize, fanOut, 1.0f);
        var expected = Matrix<float>.Build.Dense(batchSize, fanIn);
        sut.Backward(delta).CopyTo(expected);

        for (var i = 0; i < 20; i++)
        {
            var actual = sut.Backward(delta);
            Assert.AreEqual(expected, actual);
        }
    }

    // currently it is not possible to gradient check layers without weights and gradients
    [Ignore]
    [TestMethod]
    public void MaxPool2DLayer_GradientCheck_BatchSize_1()
    {
        const int inputWidth = 20;
        const int inputHeight = 20;
        const int inputDepth = 2;

        const int batchSize = 1;

        // Create layer
        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height);
        GradientCheckTools.CheckLayer(sut, inputWidth, inputHeight, inputDepth, batchSize,
            1e-4f, new Random(21));
    }

    // currently it is not possible to gradient check layers without weights and gradients
    [Ignore]
    [TestMethod]
    public void MaxPool2DLayer_GradientCheck_BatchSize_10()
    {
        const int inputWidth = 20;
        const int inputHeight = 20;
        const int inputDepth = 2;

        const int batchSize = 11;

        // Create layer
        const int width = 2;
        const int height = 2;
        var sut = new MaxPool2DLayer(width, height);
        GradientCheckTools.CheckLayer(sut, inputWidth, inputHeight, inputDepth, batchSize,
            1e-4f, new Random(21));
    }
}
