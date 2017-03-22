using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Layers;
using System.Diagnostics;
using SharpLearning.Neural.Initializations;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Providers.DotNetOp;
using SharpLearning.Containers.Tensors;
using System.Linq;
using MathNet.Numerics;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Test.Providers.DotNetOp
{
    [TestClass]
    public class MaxPoolTest
    {
        [TestMethod]
        public void MaxPool_Forward()
        {
            const int batchSize = 1;
            const int inputDepth = 2;
            const int inputHeight = 4;
            const int inputWidth = 4;

            var descriptor = new MaxPool2DDescriptor(2, 2, 2, 2, 0, 0);

            var outputC = inputDepth;
            var outputW = ConvUtils.GetFilterGridLength(inputWidth, descriptor.PoolW, 
                descriptor.StrideW, descriptor.PadW, BorderMode.Undefined);

            var outputH = ConvUtils.GetFilterGridLength(inputHeight, descriptor.PoolH, 
                descriptor.StrideH, descriptor.PadH, BorderMode.Undefined);

            var inputData = new double[] { 3, 0, 0, 6, 0, 2, 3, 0, 0, 8, 10, 0, 4, 6, 0, 7, 4, 0, 2, 0, 0, 8, 3, 5, 10, 0, 12, 0, 6, 5, 3, 2 };
            var input = Variable.Create(batchSize, inputDepth, inputHeight, inputWidth);
            var output = Variable.Create(batchSize, outputC, outputH, outputW);
            var fanOut = output.DimensionOffSets[0];

            var switchX = Enumerable.Range(0, batchSize).Select(v => new int[fanOut]).ToArray();
            var switchY = Enumerable.Range(0, batchSize).Select(v => new int[fanOut]).ToArray();

            var executor = new Executor();
            executor.AssignTensor(input, inputData);

            MaxPool2D.Forward(input, output, descriptor,
                switchX, switchY, executor);

            var actual = executor.GetTensor(output);

            var expected = Tensor<double>.Build(new double[] { 3, 6, 8, 10, 8, 5, 10, 12 }, batchSize, outputC, outputH, outputW);
            Assert.AreEqual(expected, actual);
        } 
    }
}
