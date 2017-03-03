using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Layers;
using System.Diagnostics;
using SharpLearning.Neural.Initializations;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Providers.DotNetOp;
using SharpLearning.Containers.Tensors;
using System.Linq;

namespace SharpLearning.Neural.Test.Providers.DotNetOp
{
    [TestClass]
    public class BatchNormalizationTest
    {
        [TestMethod]
        public void BatchNormalization_Forward_Prototype_Timing()
        {
            const int width = 25;
            const int height = 25;
            const int depth = 32;
            const int batchSize = 128;

            var iterations = 10;

            var timer = new Stopwatch();
            var ellapsed = RunCurrentBatchNorm(iterations, timer,
                width, height, depth, batchSize);
            Trace.WriteLine($"Current: {ellapsed}");

            timer.Reset();
            ellapsed = RunDotNetBatchNorm(iterations, timer,
                width, height, depth, batchSize);
            Trace.WriteLine($"DotNet: {ellapsed}");

            Assert.IsFalse(true);
        }

        double RunDotNetBatchNorm(int iterations, Stopwatch timer,
            int width, int height, int depth, int batchSize)
        {
            var fanIn = width * height * depth;

            var input = Tensor<float>.CreateRowMajor(width, height, depth, batchSize);
            var output = Tensor<float>.CreateRowMajor(width, height, depth, batchSize);

            var scale = Tensor<float>.CreateRowMajor(fanIn);
            var bias = Tensor<float>.CreateRowMajor(fanIn);

            var batchMeans = new float[depth];
            var batchVars = new float[depth];

            var MovingAverageMeans = new float[depth];
            var MovingAverageVariance = Enumerable.Range(0, depth).Select(v => 1.0f).ToArray();

            for (int i = 0; i < iterations; i++)
            {
                timer.Start();
                BatchNormalization.Forward(input, scale, bias, batchMeans, batchVars,
                    MovingAverageMeans, MovingAverageVariance, output, true);
                timer.Stop();
            }

            return timer.ElapsedMilliseconds / (double)iterations;
        }

        double RunCurrentBatchNorm(int iterations, Stopwatch timer,
            int width, int height, int depth, int batchSize)
        {           
            var sut = new BatchNormalizationLayer();

            sut.Initialize(width, height, depth, batchSize, 
                Initialization.GlorotUniform, new Random(232));

            var fanIn = width * height * depth;
            var input = Matrix<float>.Build.Dense(batchSize, fanIn);

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
