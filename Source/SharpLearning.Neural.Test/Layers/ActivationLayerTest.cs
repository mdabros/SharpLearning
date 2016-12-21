using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Layers;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Neural.Activations;

namespace SharpLearning.Neural.Test.Layers
{
    [TestClass]
    public class ActivationLayerTest
    {
        [TestMethod]
        public void ActivationLayer_ReLu_GradientCheck()
        {
            const int fanIn = 5;
            const int batchSize = 10;

            var sut = new ActivationLayer(Activation.Relu);
            GradientCheckTools.CheckLayer(sut, fanIn, 1, 1, batchSize, 1e-4f, new Random(21));
        }

        [TestMethod]
        public void ActivationLayer_CopyLayerForPredictionModel()
        {
            var batchSize = 1;
            var random = new Random(232);

            var sut = new ActivationLayer(Activation.Relu);
            sut.Initialize(3, 3, 1, batchSize, random);

            var layers = new List<ILayer>();
            sut.CopyLayerForPredictionModel(layers);

            var actual = (ActivationLayer)layers.Single();

            Assert.AreEqual(sut.Width, actual.Width);
            Assert.AreEqual(sut.Height, actual.Height);
            Assert.AreEqual(sut.Depth, actual.Depth);

            Assert.AreEqual(sut.ActivationFunc, actual.ActivationFunc);

            Assert.AreEqual(sut.OutputActivations.RowCount, actual.OutputActivations.RowCount);
            Assert.AreEqual(sut.OutputActivations.ColumnCount, actual.OutputActivations.ColumnCount);

            Assert.AreEqual(sut.ActivationDerivative.RowCount, actual.ActivationDerivative.RowCount);
            Assert.AreEqual(sut.ActivationDerivative.ColumnCount, actual.ActivationDerivative.ColumnCount);
        }
    }
}
