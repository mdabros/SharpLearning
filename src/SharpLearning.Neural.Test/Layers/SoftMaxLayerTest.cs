using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.Neural.Test.Layers
{
    [TestClass]
    public class SoftMaxLayerTest
    {
        [TestMethod]
        public void SoftMaxLayer_CopyLayerForPredictionModel()
        {
            var batchSize = 1;
            var width = 28;
            var height = 28;
            var depth = 3;
            var numberOfClasses = 10;

            var sut = new SoftMaxLayer(numberOfClasses);
            sut.Initialize(width, height, depth, batchSize, Initialization.GlorotUniform, new Random(232));

            var layers = new List<ILayer>();
            sut.CopyLayerForPredictionModel(layers);

            var actual = (SoftMaxLayer)layers.Single();

            Assert.AreEqual(sut.Width, actual.Width);
            Assert.AreEqual(sut.NumberOfClasses, actual.NumberOfClasses);
        }

        [TestMethod]
        public void SoftMaxLayer_Forward()
        {
            var batchSize = 1;
            var width = 28;
            var height = 28;
            var depth = 3;
            var numberOfClasses = 10;
            var random = new Random(232);

            var sut = new SoftMaxLayer(numberOfClasses);
            sut.Initialize(width, height, depth, batchSize, Initialization.GlorotUniform, random);

            var input = Matrix<float>.Build.Random(batchSize, numberOfClasses, random.Next());
            var actual = sut.Forward(input);

            Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

            var expected = Matrix<float>.Build.Dense(batchSize, numberOfClasses, new float[] { 0.06976377f, 0.1327717f, 0.02337802f, 0.3784489f, 0.0777365f, 0.05847027f, 0.1072708f, 0.0503228f, 0.0624512f, 0.03938601f });
            MatrixAsserts.AreEqual(expected, actual);
        }

        [TestMethod]
        public void SoftMaxLayer_Backward()
        {
            var batchSize = 1;
            var width = 28;
            var height = 28;
            var depth = 3;
            var numberOfClasses = 10;
            var random = new Random(232);

            var sut = new SoftMaxLayer(numberOfClasses);
            sut.Initialize(width, height, depth, batchSize, Initialization.GlorotUniform, random);

            var input = Matrix<float>.Build.Random(batchSize, numberOfClasses, random.Next());
            sut.Forward(input);

            var delta = Matrix<float>.Build.Random(batchSize, numberOfClasses, random.Next());
            var actual = sut.Backward(delta);

            Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

            var expected = Matrix<float>.Build.Dense(batchSize, numberOfClasses, new float[] { -0.3891016f, -0.6150756f, 0.0618184f, -0.2334358f, 1.544145f, -1.01483f, 0.6160479f, 0.3225261f, -1.007966f, -0.1111263f });
            MatrixAsserts.AreEqual(expected, actual);
        }
    }
}
