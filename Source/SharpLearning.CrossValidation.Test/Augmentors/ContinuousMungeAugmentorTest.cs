using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.Augmentators;
using System;

namespace SharpLearning.CrossValidation.Test.Augmentors
{
    [TestClass]
    public class ContinuousMungeAugmentorTest
    {
        [TestMethod]
        public void ContinuousMunchAugmentor_Augment()
        {
            var random = new Random(2342);
            var data = new F64Matrix(10, 2);
            data.Map(() => random.NextDouble());

            var sut = new ContinuousMungeAugmentator(0.5, 1.0);
            var actual = sut.Agument(data);

            var expected = new F64Matrix(new double[] { 0.246329100917247, 0.00372775551105279, 0.797896387892727, 0.441504622549519, 0.585635684703307, 0.227548045212192, 0.254812818139239, 0.387049265851755, 0.722062122878647, 0.888775135804329, 0.714802723710799, 0.900545311114073, 0.643038906922116, 0.907352483788204, 0.316260776164132, 0.658306110025526, 0.343312428492733, 0.0337710422620974, 0.0760259759035082, 0.148426381940221 },
                10, 2);

            Assert.AreNotEqual(data, actual);
            Assert.AreEqual(expected.GetNumberOfRows(), actual.GetNumberOfRows());
            Assert.AreEqual(expected.GetNumberOfColumns(), actual.GetNumberOfColumns());

            var expectedData = expected.Data();
            var actualData = expected.Data();

            for (int i = 0; i < expectedData.Length; i++)
            {
                Assert.AreEqual(expectedData[i], actualData[i], 0.00001);
            }
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ContinuousMunchAugmentor_Constructor_Probability_Too_Low()
        {
            new ContinuousMungeAugmentator(-0.1, 1.0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ContinuousMunchAugmentor_Constructor_Probability_Too_High()
        {
            new ContinuousMungeAugmentator(1.1, 1.0);
        }
    }
}
