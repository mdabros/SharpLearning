using System;
using System.Collections.Generic;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Layers;

namespace SharpLearning.Neural.Test.Layers;

[TestClass]
public class DropoutLayerTest
{
    [TestMethod]
    public void DropoutLayer_CopyLayerForPredictionModel()
    {
        var batchSize = 1;

        var sut = new DropoutLayer(0.5);
        sut.Initialize(5, 1, 1, batchSize, Initialization.GlorotUniform, new Random(233));

        var layers = new List<ILayer>();
        sut.CopyLayerForPredictionModel(layers);

        Assert.IsTrue(layers.Count == 0);
    }

    [TestMethod]
    public void DropoutLayer_Forward()
    {
        const int fanIn = 5;
        var batchSize = 1;
        var random = new Random(232);

        var sut = new DropoutLayer(0.5);
        sut.Initialize(5, 1, 1, batchSize, Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        var actual = sut.Forward(input);

        Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

        var expected = Matrix<float>.Build.Dense(batchSize, fanIn, [0.9177308f, 1.495695f, -0.07688076f, 0f, -2.932818f]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void DropoutLayer_Backward()
    {
        const int fanIn = 5;
        var batchSize = 1;
        var random = new Random(232);

        var sut = new DropoutLayer(0.5);
        sut.Initialize(5, 1, 1, batchSize, Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        sut.Forward(input);

        var delta = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        var actual = sut.Backward(delta);

        Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

        var expected = Matrix<float>.Build.Dense(batchSize, fanIn, [-1.676851f, -1.938897f, -1.108109f, 0f, -0.4058239f]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void DropoutLayer_MultipleBackwardsPasses()
    {
        const int fanIn = 5;
        var batchSize = 1;
        var random = new Random(232);

        var sut = new DropoutLayer(0.5);
        sut.Initialize(5, 1, 1, batchSize, Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        sut.Forward(input);

        var delta = Matrix<float>.Build.Dense(batchSize, fanIn, 1.0f);
        var expected = Matrix<float>.Build.Dense(batchSize, fanIn);
        sut.Backward(delta).CopyTo(expected);

        for (var i = 0; i < 20; i++)
        {
            var actual = sut.Backward(delta);
            Assert.AreEqual(expected, actual);
        }
    }
}
