using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Layers;

namespace SharpLearning.Neural.Test.Layers
{
    [TestClass]
    public class BatchNormalizationLayerTest
    {
        [TestMethod]
        public void BatchNormalizationLayer_CopyLayerForPredictionModel()
        {
            var batchSize = 1;

            var sut = new BatchNormalizationLayer();
            sut.Initialize(3, 3, 1, batchSize, Initialization.GlorotUniform, new Random(232));

            var layers = new List<ILayer>();
            sut.CopyLayerForPredictionModel(layers);

            var actual = (BatchNormalizationLayer)layers.Single();

            Assert.AreEqual(sut.Width, actual.Width);
            Assert.AreEqual(sut.Height, actual.Height);
            Assert.AreEqual(sut.Depth, actual.Depth);

            MatrixAsserts.AreEqual(sut.Scale, actual.Scale);
            MatrixAsserts.AreEqual(sut.Bias, actual.Bias);

            MatrixAsserts.AreEqual(sut.MovingAverageMeans, actual.MovingAverageMeans);
            MatrixAsserts.AreEqual(sut.MovingAverageVariance, actual.MovingAverageVariance);
            Assert.IsNull(actual.BatchColumnMeans);
            Assert.IsNull(actual.BatchcolumnVars);


            Assert.AreEqual(sut.OutputActivations.RowCount, actual.OutputActivations.RowCount);
            Assert.AreEqual(sut.OutputActivations.ColumnCount, actual.OutputActivations.ColumnCount);
        }

        [TestMethod]
        public void BatchNormalizationLayer_Forward()
        {
            const int fanIn = 4;
            const int batchSize = 2;

            var sut = new BatchNormalizationLayer();
            sut.Initialize(1, 1, fanIn, batchSize, Initialization.GlorotUniform, new Random(232));

            var data = new float[] { 0, 1, -1, 1, 0.5f, 1.5f, -10, 10 };
            var input = Matrix<float>.Build.Dense(batchSize, fanIn, data);

            Trace.WriteLine(input.ToString());

            var actual = sut.Forward(input);

            Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));
            Trace.WriteLine(actual);

            var expected = Matrix<float>.Build.Dense(batchSize, fanIn, new float[] { -0.999998f, 0.999998f, -0.9999995f, 0.9999995f, -0.999998f, 0.999998f, -1, 1 });
            MatrixAsserts.AreEqual(expected, actual);
        }

        [TestMethod]
        public void BatchNormalizationLayer_Forward_SpatialInput()
        {
            var batchSize = 2;

            var filterHeight = 2;
            var filterWidth = 2;
            var filterDepth = 2;

            var stride = 1;
            var padding = 0;

            var inputWidth = 3;
            var inputHeight = 3;
            var inputDepth = 3;

            var filterGridWidth = ConvUtils.GetFilterGridLength(inputWidth, filterWidth, 
                stride, padding, BorderMode.Valid);

            var filterGridHeight = ConvUtils.GetFilterGridLength(inputHeight, filterHeight, 
                stride, padding, BorderMode.Valid);

            var k = filterDepth;

            var input = new float[] { 111, 121, 112, 122, 113, 123, 114, 124, 211, 221, 212, 222, 213, 223, 214, 224 };
            var convInput = Matrix<float>.Build.Dense(2, 8, input);
            var rowWiseInput = Matrix<float>.Build.Dense(batchSize, k * filterGridWidth * filterGridHeight);

            Trace.WriteLine(convInput);

            ConvUtils.ReshapeConvolutionsToRowMajor(convInput, inputDepth, inputHeight, inputWidth, 
                filterHeight, filterWidth, padding, padding, stride, stride, 
                BorderMode.Valid, rowWiseInput);

            Trace.WriteLine(rowWiseInput);

            var sut = new BatchNormalizationLayer();
            sut.Initialize(filterGridWidth, filterGridHeight, filterDepth, batchSize, 
                Initialization.GlorotUniform, new Random(232));

            var actual = sut.Forward(rowWiseInput);

            Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));
            Trace.WriteLine(actual);

            var expected = Matrix<float>.Build.Dense(batchSize, k * filterGridWidth * filterGridHeight, new float[] { -1.0297426f, 0.9697576f, -1.00974762f, 0.9897526f, -0.9897526f, 1.00974762f, -0.9697576f, 1.0297426f, -1.0297426f, 0.9697576f, -1.00974762f, 0.9897526f, -0.9897526f, 1.00974762f, -0.9697576f, 1.0297426f });
            MatrixAsserts.AreEqual(expected, actual);
        }

        [TestMethod]
        public void BatchNormalizationLayer_Backward()
        {
            const int fanIn = 4;
            const int batchSize = 2;
            var random = new Random(232);

            var sut = new BatchNormalizationLayer();
            sut.Initialize(1, 1, fanIn, batchSize, Initialization.GlorotUniform, random);

            var data = new float[] { 0, 1, -1, 1, 0.5f, 1.5f, -10, 10 };
            var input = Matrix<float>.Build.Dense(batchSize, fanIn, data);

            Trace.WriteLine(input.ToString());
            sut.Forward(input);

            var delta = Matrix<float>.Build.Random(batchSize, fanIn, random.Next());
            var actual = sut.Backward(delta);

            Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));

            var expected = Matrix<float>.Build.Dense(batchSize, fanIn, new float[] { -2.600517E-06f, 2.615418E-06f, -1.349278E-06f, 1.349278E-06f, 1.158319E-06f, -1.150868E-06f, -5.639333E-10f, -9.261829E-10f });
            MatrixAsserts.AreEqual(expected, actual);
        }

        [TestMethod]
        public void BatchNormalizationLayer_GradientCheck_BatchSize_1()
        {
            const int fanIn = 5;
            const int batchSize = 1;

            var sut = new BatchNormalizationLayer();
            GradientCheckTools.CheckLayer(sut, 1, 1, fanIn, batchSize, 1e-4f, new Random(21));
        }

        [TestMethod]
        public void BatchNormalizationLayer_GradientCheck_BatchSize_10()
        {
            const int fanIn = 5;
            const int batchSize = 10;

            var sut = new BatchNormalizationLayer();
            GradientCheckTools.CheckLayer(sut, 1, 1, fanIn, batchSize, 1e-4f, new Random(21));
        }
    }
}
