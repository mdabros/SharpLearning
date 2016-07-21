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

        [TestMethod]
        public void MinMaxTransformer_Transform_Vector()
        {
            var sut = new MinMaxTransformer(-1.0, 1.0);
            var matrix = new F64Matrix(new double[] { -10, 0, 10,
                                                       10, 0, -10,
                                                      -10, 0, 10}, 3, 3);
            // Create transformation
            sut.Transform(matrix);

            // use vector transform on each row
            var actual = new F64Matrix(3, 3);
            for (int i = 0; i < actual.GetNumberOfRows(); i++)
            {
                var row = sut.Transform(matrix.GetRow(i));
                for (int j = 0; j < actual.GetNumberOfColumns(); j++)
                {
                    actual[i, j] = row[j];
                }
            }

            var expected = new F64Matrix(new double[] { -1, -1, 1,
                                                         1, -1, -1,
                                                        -1, -1, 1}, 3, 3);

            Assert.AreEqual(expected, actual);
        }
    }
}
