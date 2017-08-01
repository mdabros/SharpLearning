using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Neural.Providers.DotNetOp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Neural.Test.Providers.DotNetOp
{
    [TestClass]
    public class ReluTest
    {
        [TestMethod]
        public void Relu_Forward()
        {
            var input = Tensor<double>.Build(new double[] { -10, -1, 0, 1, 10 }, 5);
            var actual = Tensor<double>.Build(input.ElementCount);

            Relu.Forward(input, actual);

            var expected = Tensor<double>.Build(new double[] { 0, 0, 0, 1, 10 }, 5);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void Relu_Backward()
        {
            var output = Tensor<double>.Build(new double[] { 0, 0, 0, 1, 10 }, 5);
            var outputGradient = Tensor<double>.Build(new double[] { -10, -1, 0, 11, 101 }, 5);

            var actual = Tensor<double>.Build(output.ElementCount);

            Relu.Backward(output, outputGradient, actual);

            var expected = Tensor<double>.Build(new double[] { 0, 0, 0, 11, 101 }, 5);

            Assert.AreEqual(expected, actual);
        }
    }
}
