using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.FeatureTransformations.MatrixTransforms;

namespace SharpLearning.FeatureTransformations.Test.MatrixTransforms;

[TestClass]
public class MeanZeroFeatureTransformerTest
{
    [TestMethod]
    public void FeatureNormalizationTransformer_Transform_Matrix()
    {
        var sut = new MeanZeroFeatureTransformer();
        var matrix = new F64Matrix(new double[] { 123, 12,
                                                  41, 120,
                                                  124, 122 }, 3, 2);
        var actual = new F64Matrix(3, 2);

        sut.Transform(matrix, actual);

        var expected = new F64Matrix(new double[] { 27, -72.666666666666671,
                                                   -55, 35.333333333333329,
                                                    28, 37.333333333333329 }, 3, 2);

        Assert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void FeatureNormalizationTransformer_Transform_Vector()
    {
        var sut = new MeanZeroFeatureTransformer();
        var matrix = new F64Matrix(new double[] { 123, 12,
                                                  41, 120,
                                                  124, 122 }, 3, 2);
        // Create transformation
        sut.Transform(matrix);

        // use vector transform on each row
        var actual = new F64Matrix(3, 2);
        for (var i = 0; i < actual.RowCount; i++)
        {
            var row = sut.Transform(matrix.Row(i));
            for (var j = 0; j < actual.ColumnCount; j++)
            {
                actual[i, j] = row[j];
            }
        }

        var expected = new F64Matrix(new double[] { 27, -72.666666666666671,
                                                   -55, 35.333333333333329,
                                                    28, 37.333333333333329 }, 3, 2);
        Assert.AreEqual(expected, actual);
    }
}
