using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Layers;

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

            var sut = new ActivationLayer(Activation.Relu);
            sut.Initialize(3, 3, 1, batchSize, Initialization.GlorotUniform, new Random(232));

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
