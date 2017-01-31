using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.Neural.Test.Layers
{
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
            sut.Initialize(5, 1, 1, batchSize, random);

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
            sut.Initialize(5, 1, 1, batchSize, random);

            var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
            var actual = sut.Forward(input);

            Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

            var expected = Matrix<float>.Build.Dense(batchSize, neuronCount, new float[] { 0.9332361f, 0.4143196f, 0.4015771f, -0.9911515f, -0.4725787f, 0.07631265f });
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
            sut.Initialize(5, 1, 1, batchSize, random);

            var input = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
            sut.Forward(input);

            var delta = Matrix<float>.Build.Random(batchSize, neuronCount, random.Next());
            var actual = sut.Backward(delta);

            Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

            var expected = Matrix<float>.Build.Dense(batchSize, fanIn, new float[] { 0.001648388f, -0.2465896f, -0.6055009f, -0.01361072f, 0.434257f, -0.6961878f, -0.6534721f, 0.1021654f, -0.5873953f, -1.138367f });
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
            sut.Initialize(5, 1, 1, batchSize, random);

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
            sut.Initialize(5, 1, 1, batchSize, random);

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
}
