using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Loss;

namespace SharpLearning.Neural.Test.Loss;

[TestClass]
public class AccuracyLossTest
{
    [TestMethod]
    public void Accuracy_Loss_1()
    {
        var sut = new AccuracyLoss();
        var targets = Matrix<float>.Build.Dense(3, 2, [1, 0, 1, 0, 1, 0]);
        var predictions = Matrix<float>.Build.Dense(3, 2, [0.9f, 0.8f, 0.7f, 0.1f, 0.2f, 0.3f]);

        var actual = sut.Loss(targets, predictions);
        Assert.AreEqual(0.3333, actual, 0.001);
    }

    [TestMethod]
    public void Accuracy_Loss_2()
    {
        var sut = new AccuracyLoss();
        var targets = Matrix<float>.Build.Dense(3, 2, [1, 0, 1, 0, 1, 0]);
        var predictions = Matrix<float>.Build.Dense(3, 2, [0.9f, 0.8f, 0.3f, 0.1f, 0.2f, 0.7f]);

        var actual = sut.Loss(targets, predictions);
        Assert.AreEqual(0.6666, actual, 0.001);
    }
}
