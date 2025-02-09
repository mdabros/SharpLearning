using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Layers;

namespace SharpLearning.Neural.Test.Layers;

/// <summary>
/// Summary description for SquaredErrorRegressionLayerTest
/// </summary>
[TestClass]
public class SquaredErrorRegressionLayerTest
{
    [TestMethod]
    public void SquaredErrorRegressionLayer_CopyLayerForPredictionModel()
    {
        var batchSize = 1;
        var width = 28;
        var height = 28;
        var depth = 3;
        var numberOfTargets = 10;
        var random = new Random(232);

        var sut = new SquaredErrorRegressionLayer(numberOfTargets);
        sut.Initialize(width, height, depth, batchSize, Initialization.GlorotUniform, random);

        var layers = new List<ILayer>();
        sut.CopyLayerForPredictionModel(layers);

        var actual = (SquaredErrorRegressionLayer)layers.Single();

        Assert.AreEqual(sut.Width, actual.Width);
        Assert.AreEqual(sut.NumberOfTargets, actual.NumberOfTargets);
    }

    [TestMethod]
    public void SquaredErrorRegressionLayer_Forward()
    {
        var batchSize = 1;
        var width = 28;
        var height = 28;
        var depth = 3;
        var numberOfClasses = 10;
        var random = new Random(232);

        var sut = new SquaredErrorRegressionLayer(numberOfClasses);
        sut.Initialize(width, height, depth, batchSize, Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, numberOfClasses, random.Next());
        var actual = sut.Forward(input);

        Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

        var expected = Matrix<float>.Build.Dense(batchSize, numberOfClasses, [0.1234713f, 0.7669879f, -0.9698473f, 1.814438f, 0.2316814f, -0.05312517f, 0.5537131f, -0.2031853f, 0.01274186f, -0.4482329f]);
        MatrixAsserts.AreEqual(expected, actual);
    }

    [TestMethod]
    public void SquaredErrorRegressionLayer_Backward()
    {
        var batchSize = 1;
        var width = 28;
        var height = 28;
        var depth = 3;
        var numberOfClasses = 10;
        var random = new Random(232);

        var sut = new SquaredErrorRegressionLayer(numberOfClasses);
        sut.Initialize(width, height, depth, batchSize, Initialization.GlorotUniform, random);

        var input = Matrix<float>.Build.Random(batchSize, numberOfClasses, random.Next());
        sut.Forward(input);

        var delta = Matrix<float>.Build.Random(batchSize, numberOfClasses, random.Next());
        var actual = sut.Backward(delta);

        Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

        var expected = Matrix<float>.Build.Dense(batchSize, numberOfClasses, [-0.3353941f, 0.0191406f, -0.9314069f, 1.202553f, 1.69809f, -1.126425f, 1.06249f, 0.06901796f, -1.057676f, -0.5987452f]);
        MatrixAsserts.AreEqual(expected, actual);
    }
}
