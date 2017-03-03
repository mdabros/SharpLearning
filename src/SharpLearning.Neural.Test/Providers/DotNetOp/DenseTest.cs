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
    public class DenseTest
    {
        [TestMethod]
        public void DenseTest_Forward_Prototype_Timing()
        {
            const int width = 25;
            const int height = 25;
            const int depth = 1;
            const int batchSize = 10;
            const int units = 800;

            var iterations = 10;

            var timer = new Stopwatch();
            var ellapsed = RunCurrent(iterations, timer,
                width, height, depth, batchSize, units);
            Trace.WriteLine($"Current: {ellapsed}");

            timer.Reset();
            ellapsed = RunDotNet(iterations, timer,
                width, height, depth, batchSize, units);
            Trace.WriteLine($"DotNet: {ellapsed}");

            Assert.IsFalse(true);
        }

        double RunDotNet(int iterations, Stopwatch timer,
            int width, int height, int depth, int batchSize, int units)
        {
            var fanIn = width * height * depth;
            var fanOut = units;

            var weights = Tensor<float>.CreateRowMajor(fanIn, fanOut);
            var bias = Tensor<float>.CreateRowMajor(fanOut);

            var input = Tensor<float>.CreateRowMajor(width, height, depth, batchSize);
            var output = Tensor<float>.CreateRowMajor(1, 1, fanOut, batchSize);

            for (int i = 0; i < iterations; i++)
            {
                timer.Start();
                Dense.Forward(input, weights, bias, output);
                timer.Stop();
            }

            return timer.ElapsedMilliseconds / (double)iterations;
        }

        double RunCurrent(int iterations, Stopwatch timer,
            int width, int height, int depth, int batchSize, int units)
        {           
            var sut = new DenseLayer(units);

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
