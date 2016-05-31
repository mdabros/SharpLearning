using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.FeatureTransformations.MatrixTransforms;

namespace SharpLearning.FeatureTransformations.Test.MatrixTransforms
{
    [TestClass]
    public class MinMaxTransformerTest
    {
        [TestMethod]
        public void MinMaxTransformer_Transform()
        {
            var sut = new MinMaxTransformer(-1.0, 1.0);
            var matrix = new F64Matrix(new double[] { -10, 0, 10,
                                                       10, 0, -10,
                                                      -10, 0, 10}, 3, 3);
            var actual = sut.Transform(matrix);

            var expected = new F64Matrix(new double[] { -1, -1, 1,
                                                         1, -1, -1,
                                                        -1, -1, 1}, 3, 3);

            Assert.AreEqual(expected, actual);
        }
    }
}
