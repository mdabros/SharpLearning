using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Layers;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.Diagnostics;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.Test.Layers
{
    [TestClass]
    public class SvmLayerTest
    {
        [TestMethod]
        public void SvmLayer_CopyLayerForPredictionModel()
        {
            var batchSize = 1;
            var width = 28;
            var height = 28;
            var depth = 3;
            var numberOfClasses = 10;

            var sut = new SvmLayer(numberOfClasses);
            sut.Initialize(width, height, depth, batchSize, Initialization.GlorotUniform, new Random(232));

            var layers = new List<ILayer>();
            sut.CopyLayerForPredictionModel(layers);

            var actual = (SvmLayer)layers.Single();

            Assert.AreEqual(sut.Width, actual.Width);
            Assert.AreEqual(sut.NumberOfClasses, actual.NumberOfClasses);
        }

        [TestMethod]
        public void SvmLayer_Forward()
        {
            var batchSize = 1;
            var width = 28;
            var height = 28;
            var depth = 3;
            var numberOfClasses = 10;
            var random = new Random(232);

            var sut = new SvmLayer(numberOfClasses);
            sut.Initialize(width, height, depth, batchSize, Initialization.GlorotUniform, random);

            var input = Matrix<float>.Build.Random(batchSize, numberOfClasses, random.Next());
            var actual = sut.Forward(input);

            Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

            var expected = Matrix<float>.Build.Dense(batchSize, numberOfClasses, new float[] { 0.1234713f, 0.7669879f, -0.9698473f, 1.814438f, 0.2316814f, -0.05312517f, 0.5537131f, -0.2031853f, 0.01274186f, -0.4482329f });
            MatrixAsserts.AreEqual(expected, actual);
        }

        [TestMethod]
        public void SvmLayer_Backward()
        {
            var batchSize = 1;
            var width = 28;
            var height = 28;
            var depth = 3;
            var numberOfClasses = 10;
            var random = new Random(232);

            var sut = new SvmLayer(numberOfClasses);
            sut.Initialize(width, height, depth, batchSize, Initialization.GlorotUniform, random);

            var input = Matrix<float>.Build.Random(batchSize, numberOfClasses, random.Next());
            sut.Forward(input);

            var delta = Matrix<float>.Build.Random(batchSize, numberOfClasses, random.Next());
            var actual = sut.Backward(delta);

            Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

            var expected = Matrix<float>.Build.Dense(batchSize, numberOfClasses, new float[] { 1f, 1f, 1f, 1f, 1f, -9f, 1f, 1f, 1f, 1f });
            MatrixAsserts.AreEqual(expected, actual);
        }
    }
}
