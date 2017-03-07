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
        public void MaxPoolTest_Forward_Prototype_Timing()
        {
            const int width = 32;
            const int height = 32;
            const int depth = 3;
            const int batchSize = 128;
            const int units = 256;

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

            var switches = new int[batchSize * fanIn];

            var input = Tensor<float>.CreateRowMajor(batchSize, depth, height, width);
            var output = Tensor<float>.CreateRowMajor(batchSize, outputC, outputH, outputW);
            
            MaxPool.Forward(input, poolHeight, poolWidth, 
                strideH, strideW, padH, padW, switches, output);

            for (int i = 0; i < iterations; i++)
            {
                timer.Start();
                MaxPool.Forward(input, poolHeight, poolWidth,
                    strideH, strideW, padH, padW, switches, output);
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
