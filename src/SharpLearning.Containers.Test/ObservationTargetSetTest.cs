using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers.Test
{
    [TestClass]
    public class ObservationTargetSetTest
    {
        [TestMethod]
        public void ObservationTargetSet_Equals()
        {
            var observations1 = new F64Matrix(new double[] { 1, 2, 3, 4 }, 2, 2);
            var observations2 = new F64Matrix(new double[] { 4, 3, 2, 1 }, 2, 2);

            var targets1 = new double[] { 1, 2 };
            var targets2 = new double[] { 2, 1 };

            var sut = new ObservationTargetSet(observations1, targets1);
            var equal = new ObservationTargetSet(observations1, targets1);
            var notEqual1 = new ObservationTargetSet(observations2, targets1);
            var notEqual2 = new ObservationTargetSet(observations1, targets2);
            var notEqual3 = new ObservationTargetSet(observations1, targets2);

            Assert.AreEqual(sut, equal);
            Assert.AreNotEqual(sut, notEqual1);
            Assert.AreNotEqual(sut, notEqual2);
            Assert.AreNotEqual(sut, notEqual3);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ObservationTargetSet_Observations_Null()
        {
            new ObservationTargetSet(null, new double[] { 1 });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ObservationTargetSet_Targets_Null()
        {
            new ObservationTargetSet(new F64Matrix(new double[] { 1, 2, 3, 4 }, 2, 2), null);
        }

    }
}
