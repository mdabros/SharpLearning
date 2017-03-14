using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Containers.Test.Tensors
{
    [TestClass]
    public class TensorTest
    {
        [TestMethod]
        public void Tensor_SliceCopy()
        {
            var shape = new TensorShape(50, 40, 30, 20);
            var data = Enumerable.Range(0, shape.ElementCount).ToArray();
            var sut = new Tensor<int>(data, shape, DataLayout.RowMajor);

            var index = 0;
            for (int n = 0; n < sut.Dimensions[0]; n++)
            {
                for (int c = 0; c < sut.Dimensions[1]; c++)
                {
                    for (int h = 0; h < sut.Dimensions[2]; h++)
                    {
                        var wSlice = sut.SliceCopy(n)
                                        .SliceCopy(c)
                                        .SliceCopy(h).Data;

                        for (int w = 0; w < sut.Dimensions[3]; w++)
                        {
                            var actual = wSlice[w];
                            var expected = data[index];
                            Assert.AreEqual(expected, actual);
                            index++;                                 
                        }
                    }
                }
            }
        }
    }
}
