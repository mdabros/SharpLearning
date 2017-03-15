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

        [TestMethod]
        public void MaxPoolTest_Forward_Prototype_Timing()
        {
            const int width = 32;
            const int height = 32;
            const int depth = 64;
            const int batchSize = 128;

            int poolHeight = 2;
            int poolWidth = 2;

            int strideH = 2;
            int strideW = 2;

            int padH = 0;
            int padW = 0;

            var iterations = 10;

            var timer = new Stopwatch();
            var ellapsed = RunCurrent(iterations, timer,
                width, height, depth, batchSize,
                poolHeight, poolWidth, 
                strideH, strideW, padH, padW);

            Trace.WriteLine($"Current: {ellapsed}");

            timer.Reset();
            ellapsed = RunDotNet(iterations, timer,
                width, height, depth, batchSize,
                poolHeight, poolWidth,
                strideH, strideW, padH, padW);

            Trace.WriteLine($"DotNet: {ellapsed}");

            //Assert.IsFalse(true);
        }

        double RunDotNet(int iterations, Stopwatch timer,
            int width, int height, int depth, int batchSize,
            int poolHeight, int poolWidth,
            int strideH, int strideW,
            int padH, int padW)
        {
            // computed
            var outputC = depth;
            var outputW = ConvUtils.GetFilterGridLength(width, poolWidth, strideW, padW, BorderMode.Undefined);
            var outputH = ConvUtils.GetFilterGridLength(height, poolHeight, strideH, padH, BorderMode.Undefined);

            // store switches for x,y coordinates for where the max comes from, for each output neuron
            var fanIn = width * depth * height;
            var fanOut = outputC * outputW * outputH;

            var input = Tensor<float>.Build(batchSize, depth, height, width);
            var output = Tensor<float>.Build(batchSize, outputC, outputH, outputW);

            var sut = new MaxPool(poolHeight, poolWidth,
                strideH, strideW, padH, strideW,
                batchSize, outputC, outputH, outputW);

            // warmup
            sut.Forward(input, output);

            for (int i = 0; i < iterations; i++)
            {
                timer.Start();
                sut.Forward(input, output);
                timer.Stop();
            }

            return timer.ElapsedMilliseconds / (double)iterations;
        }

        double RunCurrent(int iterations, Stopwatch timer,
            int width, int height, int depth, int batchSize,
            int poolHeight, int poolWidth,
            int strideH, int strideW,
            int padH, int padW)
        {
            var sut = new MaxPool2DLayer(poolWidth, poolHeight, strideH, padW, padH);

            sut.Initialize(width, height, depth, batchSize, 
                Initialization.GlorotUniform, new Random(232));

            var fanIn = width * height * depth;
            var input = Matrix<float>.Build.Dense(batchSize, fanIn);

            sut.Forward(input);

            for (int i = 0; i < iterations; i++)
            {
                timer.Start();
                var actual = sut.Forward(input);
                timer.Stop();
            }

            return timer.ElapsedMilliseconds / (double)iterations;
        }   
    }
}
