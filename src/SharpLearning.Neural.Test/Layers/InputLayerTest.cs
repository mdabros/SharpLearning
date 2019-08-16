using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Layers;

namespace SharpLearning.Neural.Test.Layers
{
    [TestClass]
    public class InputLayerTest
    {
        [TestMethod]
        public void InputLayer_CopyLayerForPredictionModel()
        {
            var batchSize = 1;
            var width = 28;
            var height = 28;
            var depth = 3;

            var sut = new InputLayer(height, width, depth);
            sut.Initialize(1, 1, 1, batchSize, Initialization.GlorotUniform, new Random(232));

            var layers = new List<ILayer>();
            sut.CopyLayerForPredictionModel(layers);

            var actual = (InputLayer)layers.Single();

            Assert.AreEqual(sut.Width, actual.Width);
            Assert.AreEqual(sut.Height, actual.Height);
            Assert.AreEqual(sut.Depth, actual.Depth);
        }

        [TestMethod]
        public void InputLayer_Forward()
        {
            var batchSize = 1;
            var width = 28;
            var height = 28;
            var depth = 3;
            var random = new Random(232);
            var fanIn = width * height * depth;

            var sut = new InputLayer(height, width, depth);
            sut.Initialize(1, 1, 1, batchSize, Initialization.GlorotUniform, random);

            var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
            var actual = sut.Forward(input);

            var expected = input;
            MatrixAsserts.AreEqual(expected, actual);
        }

        [TestMethod]
        public void InputLayer_Backward()
        {
            var batchSize = 1;
            var width = 28;
            var height = 28;
            var depth = 3;
            var random = new Random(232);
            var fanIn = width * height * depth;

            var sut = new InputLayer(height, width, depth);
            sut.Initialize(1, 1, 1, batchSize, Initialization.GlorotUniform, random);

            var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
            sut.Forward(input);

            var delta = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
            var actual = sut.Backward(delta);

            var expected = delta;
            MatrixAsserts.AreEqual(expected, actual);
        }
    }
}
