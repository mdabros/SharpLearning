using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Loss;

namespace SharpLearning.Neural.Test.Loss;

[TestClass]
public class SquareLossTest
{
    [TestMethod]
    public void SquareLoss_Loss()
    {
        var targets = Matrix<float>.Build.Dense(6, 1, [0, 0, 0, 0, 0, 0]);
        var predictions = Matrix<float>.Build.Dense(6, 1, [0, 0, 0, 0, 0, 0]);

        var sut = new SquareLoss();
        var actual = sut.Loss(targets, predictions);

        Assert.AreEqual(0f, actual);
    }

    [TestMethod]
    public void SquareLoss_Loss_1()
    {
        var targets = Matrix<float>.Build.Dense(5, 1, [1.0f, 2.3f, 3.1f, 4.4f, 5.8f]);
        var predictions = Matrix<float>.Build.Dense(5, 1, [1.0f, 2.0f, 3.0f, 4.0f, 5.0f]);

        var sut = new SquareLoss();
        var actual = sut.Loss(targets, predictions);

        Assert.AreEqual(0.09f, actual, 0.0001);
    }

    [TestMethod]
    public void SquareLoss_Loss_Multi_Dimensional()
    {
        var targets = Matrix<float>.Build.Dense(3, 3,
            [1.0f, 2.3f, 3.1f, 4.4f, 5.8f, 1.0f, 3.5f, 2f, 5f]);

        var predictions = Matrix<float>.Build.Dense(3, 3,
            [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 3.7f, 1.6f, 5.4f]);

        var sut = new SquareLoss();
        var actual = sut.Loss(targets, predictions);

        Assert.AreEqual(0.07f, actual, 0.0001);
    }
}
