using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Test.LayersNew
{
    [TestClass]
    public class MaxPool2DLayerTest
    {
        [TestMethod]
        public void MaxPool2DLayer_GradientCheck()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(10, 3, 5, 5);
            var sut = new MaxPool2DLayer(2, 2);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayer(sut, storage, input, new Random(21));
        }

        [TestMethod]
        public void MaxPool2DLayer_Forward_1()
        {
            var storage = new NeuralNetStorage();
            var inputData = new double[] { 3, 0, 0, 6,
                                           0, 2, 3, 0,
                                           0, 8, 10, 0,
                                           4, 6, 0, 7 };

            var input = Variable.Create(1, 1, 4, 4);

            storage.AssignTensor(input, inputData);
            var sut = new MaxPool2DLayer(2, 2);
            sut.Initialize(input, storage, new Random(23));
            sut.Forward(storage);

            var expected = Tensor<double>.Build(new double[] { 3, 6, 8, 10 }, 1, 1, 2, 2);
            var actual = storage.GetTensor(sut.Output);
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void MaxPool2DLayer_Forward_2()
        {
            var storage = new NeuralNetStorage();
            var inputData = new double[] { 3, 0, 0, 6,
                                            0, 2, 3, 0,
                                            0, 8, 10, 0,
                                            4, 6, 0, 7,
                                            4, 0, 2, 0,
                                            0, 8, 3, 5,
                                            10, 0, 12, 0,
                                            6, 5, 3, 2 };

            var input = Variable.Create(2, 1, 4, 4);

            storage.AssignTensor(input, inputData);
            var sut = new MaxPool2DLayer(2, 2);
            sut.Initialize(input, storage, new Random(23));
            sut.Forward(storage);

            var expected = Tensor<double>.Build(new double[] { 3, 6, 8, 10,
                                                               8, 5, 10, 12 }, 2, 1, 2, 2);
            var actual = storage.GetTensor(sut.Output);
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void MaxPool2DLayer_Backward_1()
        {
            var storage = new NeuralNetStorage();
            var inputData = new double[] { 3, 0, 0, 6,
                                           0, 2, 3, 0,
                                           0, 8, 10, 0,
                                           4, 6, 0, 7 };

            var input = Variable.Create(1, 1, 4, 4);

            storage.AssignTensor(input, inputData);
            var sut = new MaxPool2DLayer(2, 2);
            sut.Initialize(input, storage, new Random(23));
            sut.Forward(storage);

            storage.AssignGradient(sut.Output, () => 1.0);

            sut.Backward(storage);

            var actual = storage.GetGradient(sut.Input);
            var expected = Tensor<double>.Build(new double[] { 1, 0, 0, 1,
                                                               0, 0, 0, 0,
                                                               0, 1, 1, 0,
                                                               0, 0, 0, 0 }, 1, 1, 4, 4);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void MaxPool2DLayer_Backward_2()
        {
            var storage = new NeuralNetStorage();
            var inputData = new double[] { 3, 0, 0, 6,
                                            0, 2, 3, 0,
                                            0, 8, 10, 0,
                                            4, 6, 0, 7,
                                            4, 0, 2, 0,
                                            0, 8, 3, 5,
                                            10, 0, 12, 0,
                                            6, 5, 3, 2 };

            var input = Variable.Create(2, 1, 4, 4);

            storage.AssignTensor(input, inputData);
            var sut = new MaxPool2DLayer(2, 2);
            sut.Initialize(input, storage, new Random(23));
            sut.Forward(storage);

            storage.AssignGradient(sut.Output, () => 1.0);

            sut.Backward(storage);

            var actual = storage.GetGradient(sut.Input);
            var expected = Tensor<double>.Build(new double[] { 1, 0, 0, 1,
                                                               0, 0, 0, 0,
                                                               0, 1, 1, 0,
                                                               0, 0, 0, 0,
                                                               0, 0, 0, 0,
                                                               0, 1, 0, 1,
                                                               1, 0, 1, 0,
                                                               0, 0, 0, 0 }, 2, 1, 4, 4);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void MaxPool2DLayer_MultipleForwardsPasses()
        {
            var random = new Random(32);
            var storage = new NeuralNetStorage();
            var input = Variable.Create(10, 3, 20, 20);

            storage.AssignTensor(input, () => random.NextDouble());
            var sut = new MaxPool2DLayer(2, 2);
            sut.Initialize(input, storage, new Random(23));

            sut.Forward(storage);
            var output = storage.GetTensor(sut.Output);
            var expected = Tensor<double>.Build(output.Data.ToArray(),
                output.Dimensions.ToArray());

            for (int i = 0; i < 20; i++)
            {
                sut.Forward(storage);
                var actual = storage.GetTensor(sut.Output);
                Assert.AreEqual(expected, actual);
            }
        }

        [TestMethod]
        public void MaxPool2DLayer_MultipleBackwardsPasses()
        {
            var random = new Random(232);
            var storage = new NeuralNetStorage();

            var input = Variable.Create(2, 1, 4, 4);
            storage.AssignTensor(input, () => random.Next(256));

            var sut = new MaxPool2DLayer(2, 2);
            sut.Initialize(input, storage, random);

            sut.Forward(storage);
            storage.AssignGradient(sut.Output, () => 1.0);

            sut.Backward(storage);
            var inputGradient = storage.GetGradient(sut.Input);

            var expected = Tensor<double>.Build(inputGradient.Data.ToArray(),
                inputGradient.Dimensions.ToArray());

            for (int i = 0; i < 20; i++)
            {
                sut.Backward(storage);
                var actual = storage.GetGradient(sut.Input);

                Trace.WriteLine(string.Join(",", expected.Data));
                Trace.WriteLine(string.Join(",", actual.Data));

                Assert.AreEqual(expected, actual);
            }
        }
    }
}
