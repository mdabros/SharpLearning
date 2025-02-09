using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Layers;

namespace SharpLearning.Neural.Test.Layers;

[TestClass]
public class DenseLayerTest
{
    [TestMethod]
    public void DenseLayer_CopyLayerForPredictionModel()
    {
        var batchSize = 1;
        var random = new Random(232);
        var neuronCount = 5;

        var sut = new DenseLayer(neuronCount, Activation.Undefined);
        sut.Initialize(5, 1, 1, batchSize, Initialization.GlorotUniform, random);

        var layers = new List<ILayer>();
        sut.CopyLayerForPredictionModel(layers);

        var actual = (DenseLayer)layers.Single();

        Assert.AreEqual(sut.Width, actual.Width);
        Assert.AreEqual(sut.Height, actual.Height);
        Assert.AreEqual(sut.Depth, actual.Depth);

        MatrixAsserts.AreEqual(sut.Weights, actual.Weights);
        MatrixAsserts.AreEqual(sut.Bias, actual.Bias);

        Assert.AreEqual(sut.OutputActivations.RowCount, actual.OutputActivations.RowCount);
        Assert.AreEqual(sut.OutputActivations.ColumnCount, actual.OutputActivations.ColumnCount);
    }

    [TestMethod]
    public void DenseLayer_Forward()
    {
        const int fanIn = 5;
        const int batchSize = 2;
        const int neuronCount = 3;
        var random = new Random(232);

        var sut = new DenseLayer(neuronCount, Activation.Undefined);
        sut.Initialize(5, 1, 1, batchSize, Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        var actual = sut.Forward(input);

        Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

        var expected = Matrix<float>.Build.Dense(batchSize, neuronCount, new float[] { 0.9898463f, 0.4394523f, 0.4259368f, -1.051275f, -0.5012454f, 0.08094172f });
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void DenseLayer_Backward()
    {
        const int fanIn = 5;
        const int batchSize = 2;
        const int neuronCount = 3;
        var random = new Random(232);

        var sut = new DenseLayer(neuronCount, Activation.Undefined);
        sut.Initialize(5, 1, 1, batchSize, Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        sut.Forward(input);

        var delta = Matrix<float>.Build.Random(batchSize, neuronCount, random.Next());
        var actual = sut.Backward(delta);

        Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

        var expected = Matrix<float>.Build.Dense(batchSize, fanIn, new float[] { 0.001748383f, -0.2615477f, -0.6422306f, -0.01443626f, 0.4605991f, -0.7384186f, -0.6931117f, 0.1083627f, -0.6230267f, -1.20742f });
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void DenseLayer_MultipleForwardsPasses()
    {
        const int fanIn = 5;
        const int batchSize = 2;
        const int neuronCount = 3;
        var random = new Random(232);

        var sut = new DenseLayer(neuronCount, Activation.Undefined);
        sut.Initialize(5, 1, 1, batchSize, Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());

        var expected = Matrix<float>.Build.Dense(batchSize, neuronCount);
        sut.Forward(input).CopyTo(expected);

        for (int i = 0; i < 20; i++)
        {
            var actual = sut.Forward(input);

            Assert.AreEqual(expected, actual);
        }
    }

    [TestMethod]
    public void DenseLayer_MultipleBackwardsPasses()
    {
        const int fanIn = 5;
        const int batchSize = 2;
        const int neuronCount = 3;
        var random = new Random(232);

        var sut = new DenseLayer(neuronCount, Activation.Undefined);
        sut.Initialize(5, 1, 1, batchSize, Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
        sut.Forward(input);

        var delta = Matrix<float>.Build.Dense(batchSize, neuronCount, 1.0f);
        var expected = Matrix<float>.Build.Dense(batchSize, fanIn);
        sut.Backward(delta).CopyTo(expected);

        for (int i = 0; i < 20; i++)
        {
            var actual = sut.Backward(delta);
            Assert.AreEqual(expected, actual);
        }
    }

    [TestMethod]
    public void DenseLayer_GradientCheck_BatchSize_1()
    {
        const int fanIn = 5;
        const int batchSize = 1;
        const int neuronCount = 3;

        var sut = new DenseLayer(neuronCount, Activation.Undefined);
        GradientCheckTools.CheckLayer(sut, fanIn, 1, 1, batchSize, 1e-4f, new Random(21));
    }

    [TestMethod]
    public void DenseLayer_GradientCheck_BatchSize_10()
    {
        const int fanIn = 5;
        const int batchSize = 10;
        const int neuronCount = 3;

        var sut = new DenseLayer(neuronCount, Activation.Undefined);
        GradientCheckTools.CheckLayer(sut, fanIn, 1, 1, batchSize, 1e-4f, new Random(21));
    }
}
