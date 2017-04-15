using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Test.LayersNew
{
    [TestClass]
    public class Conv2DLayerTest
    {
        [TestMethod]
        public void Conv2DLayer_GradientCheck()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(1, 3, 3, 3);
            var sut = new Conv2DLayer(2, 2, 2);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayer(sut, storage, input, new Random(21));
        }

        [TestMethod]
        public void Conv2DLayer_GradientCheck_Batch_5()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(5, 3, 3, 3);
            var sut = new Conv2DLayer(2, 2, 2);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayer(sut, storage, input, new Random(21));
        }

        [TestMethod]
        public void Conv2DLayer_ParameterGradientCheck()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(1, 3, 3, 3);
            var sut = new Conv2DLayer(2, 2, 2);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayerParameters(sut, storage, input, new Random(21));
        }

        [TestMethod]
        public void Conv2DLayer_ParameterGradientCheck_Batch_5()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(5, 3, 3, 3);
            var sut = new Conv2DLayer(2, 2, 2);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayerParameters(sut, storage, input, new Random(21));
        }

        [TestMethod]
        public void Conv2DLayer_Forward_Simple()
        {
            // example from: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html
            var storage = new NeuralNetStorage();
            var inputData = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var input = Variable.Create(1, 1, 3, 3);

            storage.AssignTensor(input, inputData);
            var sut = new Conv2DLayer(1, 2, 2);
            sut.Initialize(input, storage, new Random(23));
            storage.AssignTensor(sut.Weights, new double[] { 4, 3, 2, 1 }); // weights reversed.
            sut.Forward(storage);

            var expectedConv = Tensor<double>.Build(new double[] { 23, 33, 53, 63 }, 1, 1, 2, 2);
            var actualConv = storage.GetTensor(sut.Output);
            Assert.AreEqual(expectedConv, actualConv);
        }


        [TestMethod]
        public void Conv2DLayer_Forward_Backward()
        {
            var storage = new NeuralNetStorage();
            var random = new Random();

            var input = Variable.Create(5, 1, 10, 10);
            storage.AssignTensor(input, () => random.Next(256));

            var sut = new Conv2DLayer(20, 3, 3, 1, 1, 1, 1);
            sut.Initialize(input, storage, new Random());

            sut.Forward(storage);
            Trace.WriteLine(string.Join(",", storage.GetTensor(sut.Output).Data));

            storage.AssignGradient(sut.Output, () => 1);
            sut.Backward(storage);
            Trace.WriteLine(string.Join(",", storage.GetGradient(sut.Input).Data));
        }
    }
}
