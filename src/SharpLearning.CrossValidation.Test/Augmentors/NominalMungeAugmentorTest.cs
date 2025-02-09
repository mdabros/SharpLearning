using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.Augmentators;

namespace SharpLearning.CrossValidation.Test.Augmentors;

[TestClass]
public class NominalMungeAugmentorTest
{
    [TestMethod]
    public void NominalMunchAugmentor_Augment()
    {
        var random = new Random(2342);
        var data = new F64Matrix(10, 2);
        data.Map(() => random.Next(2));

        var sut = new NominalMungeAugmentator(0.5);
        var actual = sut.Agument(data);

        var expected = new F64Matrix([0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            10, 2);

        Assert.AreNotEqual(data, actual);
        Assert.AreEqual(expected.RowCount, actual.RowCount);
        Assert.AreEqual(expected.ColumnCount, actual.ColumnCount);

        var expectedData = expected.Data();
        var actualData = expected.Data();

        for (var i = 0; i < expectedData.Length; i++)
        {
            Assert.AreEqual(expectedData[i], actualData[i], 0.00001);
        }
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void NominalMunchAugmentor_Constructor_Probability_Too_Low()
    {
        new NominalMungeAugmentator(-0.1);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void NominalMunchAugmentor_Constructor_Probability_Too_High()
    {
        new NominalMungeAugmentator(1.1);
    }
}
