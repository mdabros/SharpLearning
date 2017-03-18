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

namespace SharpLearning.Neural.Test.Providers.DotNetOp
{
    [TestClass]
    public class MaxPoolTest
    {
        [TestMethod]
        public void MaxPool_Forward()
        {
            const int inputWidth = 4;
            const int inputHeight = 4;
            const int inputDepth = 2;

            const int batchSize = 1;
            const int stride = 2;
            const int pad = 0;

            // Create layer
            const int poolWidth = 2;
            const int poolHeight = 2;

            var outputC = inputDepth;
            var outputW = ConvUtils.GetFilterGridLength(inputWidth, poolWidth, stride, pad, BorderMode.Undefined);
            var outputH = ConvUtils.GetFilterGridLength(inputHeight, poolHeight, stride, pad, BorderMode.Undefined);

            var inputData = new float[] { 3, 0, 0, 6, 0, 2, 3, 0, 0, 8, 10, 0, 4, 6, 0, 7, 4, 0, 2, 0, 0, 8, 3, 5, 10, 0, 12, 0, 6, 5, 3, 2 };
            var input = Tensor<float>.Build(inputData, batchSize, inputDepth, inputHeight, inputWidth);
            var output = Tensor<float>.Build(batchSize, outputC, outputH, outputW);

            var sut = new MaxPool(poolHeight, poolWidth,
                stride, stride, pad, pad,
                batchSize, outputC, outputH, outputW);

            sut.Forward(input, output);

            var expected = Tensor<float>.Build(new float[] { 3, 6, 8, 10, 8, 5, 10, 12 }, batchSize, outputC, outputH, outputW);
            Assert.AreEqual(expected, output);
        } 
    }
}
