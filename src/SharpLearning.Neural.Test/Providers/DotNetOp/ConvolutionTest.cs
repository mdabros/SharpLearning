using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Neural.Providers.DotNetOp;

namespace SharpLearning.Neural.Test.Providers.DotNetOp
{
    [TestClass]
    public class ConvolutionTest
    {
        [TestMethod]
        public void Convolution_Im2Col_BatchSize_1()
        {
            var imData = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28 };
            var im = Tensor<double>.Build(imData, 1, 3, 3, 3);

            var actual = Tensor<double>.Build(1, 3, 4, 4);
            var convDescriptor = new Conv2DDescriptor(2, 2, 2, 1, 1, 0, 0);

            Convolution.Im2Col(im, convDescriptor, BorderMode.Valid, actual);

            Trace.WriteLine(string.Join(",", actual.Data));

            var expected = Tensor<double>.Build(new double[] { 0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 6, 7, 4, 5, 7, 8, 10, 11, 13, 14, 11, 12, 14, 15, 13, 14, 16, 17, 14, 15, 17, 18, 20, 21, 23, 24, 21, 22, 24, 25, 23, 24, 26, 27, 24, 25, 27, 28 }, 
                actual.Dimensions.ToArray());

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void Convolution_Im2Col_BatchSize_2()
        {
            var imData = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                        100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128};

            var im = Tensor<double>.Build(imData, 2, 3, 3, 3);

            var actual = Tensor<double>.Build(2, 3, 4, 4);
            var convDescriptor = new Conv2DDescriptor(2, 2, 2, 1, 1, 0, 0);

            Convolution.Im2Col(im, convDescriptor, BorderMode.Valid, actual);

            Trace.WriteLine(string.Join(",", actual.Data));

            var expected = Tensor<double>.Build(new double[] { 0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 6, 7, 4, 5, 7, 8, 10, 11, 13, 14, 11, 12, 14, 15, 13, 14, 16, 17, 14, 15, 17, 18, 20, 21, 23, 24, 21, 22, 24, 25, 23, 24, 26, 27, 24, 25, 27, 28, 100, 101, 103, 104, 101, 102, 104, 105, 103, 104, 106, 107, 104, 105, 107, 108, 110, 111, 113, 114, 111, 112, 114, 115, 113, 114, 116, 117, 114, 115, 117, 118, 120, 121, 123, 124, 121, 122, 124, 125, 123, 124, 126, 127, 124, 125, 127, 128 }, 
                actual.Dimensions.ToArray());

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void Convolution_Im2Col_ConcurrencyTest()
        {
            var random = new Random(21);
            var batchSize = 100;

            var im = Tensor<double>.Build(batchSize, 3, 3, 3);
            im.Map(v => random.Next());

            var initalExpected = Tensor<double>.Build(batchSize, 3, 4, 4);
            var convDescriptor = new Conv2DDescriptor(2, 2, 2, 1, 1, 0, 0);

            Convolution.Im2Col(im, convDescriptor, BorderMode.Valid, initalExpected);

            for (int i = 0; i < 10; i++)
            {
                var actual = Tensor<double>.Build(batchSize, 3, 4, 4);
                Convolution.Im2Col(im, convDescriptor, BorderMode.Valid, actual);
                Assert.AreEqual(initalExpected, actual);
            }
        }

        [TestMethod]
        public void Convolution_Col2Im_BatchSize_2()
        {
            var col2ImData = new double[] { 0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 6, 7, 4, 5, 7, 8, 10, 11, 13, 14, 11, 12, 14, 15, 13, 14, 16, 17, 14, 15, 17, 18, 20, 21, 23, 24, 21, 22, 24, 25, 23, 24, 26, 27, 24, 25, 27, 28, 100, 101, 103, 104, 101, 102, 104, 105, 103, 104, 106, 107, 104, 105, 107, 108, 110, 111, 113, 114, 111, 112, 114, 115, 113, 114, 116, 117, 114, 115, 117, 118, 120, 121, 123, 124, 121, 122, 124, 125, 123, 124, 126, 127, 124, 125, 127, 128 };
            var col2Im = Tensor<double>.Build(col2ImData, 2, 3, 4, 4);

            var actual = Tensor<double>.Build(2, 3, 3, 3);            
            var convDescriptor = new Conv2DDescriptor(2, 2, 2, 1, 1, 0, 0);

            Convolution.Col2Im(col2Im, convDescriptor, BorderMode.Valid, actual);

            var expected = Tensor<double>.Build(new double[] { 0, 2, 2, 6, 16, 10, 6, 14, 8, 10, 22, 12, 26, 56, 30, 16, 34, 18, 20, 42, 22, 46, 96, 50, 26, 54, 28, 100, 202, 102, 206, 416, 210, 106, 214, 108, 110, 222, 112, 226, 456, 230, 116, 234, 118, 120, 242, 122, 246, 496, 250, 126, 254, 128 },
                actual.Dimensions.ToArray());

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void Convolution_Simple()
        {
            // example from: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html
            var desc = new Conv2DDescriptor(1, 2, 2, 1, 1, 0, 0);
            var im = Tensor<double>.Build(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 1, 1, 3, 3);
            var im2Col = Tensor<double>.Build(4, 4);

            Convolution.Im2Col(im, desc, BorderMode.Valid, im2Col);
            var expectedIm2Col = Tensor<double>.Build(new double[] { 1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5,6 ,8, 9 }, 4, 4);
            Assert.AreEqual(expectedIm2Col, im2Col);

            var w = Tensor<double>.Build(new double[] { 4, 3, 2, 1 }, 1, 4);
            var actualConv = Tensor<double>.Build(1, 4);
            w.Multiply(im2Col, actualConv);
            var expectedConv = Tensor<double>.Build(new double[] { 23, 33, 53, 63 }, 1, 4);

            Assert.AreEqual(expectedConv, actualConv);
        }

        [TestMethod]
        public void SwitchDimensionOneAndTwo_FilterMajorToBatchMajor()
        {
            /// transform from tensor: [filterCount, BatchSize, GridsizeH, GridSizeW)]
            /// to tensor:             [BatchSize, filterCount, GridSizeH, GridSizeW)]
            var src = Tensor<double>.Build(new double[] { 1, 1, 1, 1, 10, 10, 10, 10,
                                                          2, 2, 2, 2, 20, 20, 20, 20,
                                                          3, 3, 3, 3, 30, 30, 30, 30, }, 3, 2, 2, 2); 
            var dst = Tensor<double>.Build(2, 3, 2, 2);

            Convolution.SwitchDimensionOneAndTwo(src, dst);

            Trace.WriteLine(string.Join(",", dst.Data));

            var expected = Tensor<double>.Build(new double[] { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                                               10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, }, 2, 3, 2, 2);
            Assert.AreEqual(expected, dst);            
        }

        [TestMethod]
        public void SwitchDimensionOneAndTwo_BatchMajorToFilterMajor()
        {
            /// transform from tensor: [BatchSize, filterCount, GridSizeH, GridSizeW)]
            /// to tensor:             [filterCount, BatchSize, GridsizeH, GridSizeW)]
            var src = Tensor<double>.Build(new double[] { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                                          10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, }, 2, 3, 2, 2);

            var dst = Tensor<double>.Build(3, 2, 2, 2);

            Convolution.SwitchDimensionOneAndTwo(src, dst);

            Trace.WriteLine(string.Join(",", dst.Data));

            var expected = Tensor<double>.Build(new double[] { 1, 1, 1, 1, 10, 10, 10, 10,
                                                               2, 2, 2, 2, 20, 20, 20, 20,
                                                               3, 3, 3, 3, 30, 30, 30, 30, }, 3, 2, 2, 2); 
            Assert.AreEqual(expected, dst);
        }
    }
}
