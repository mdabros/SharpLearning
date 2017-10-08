using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers.Test
{
    [TestClass]
    public class ChecksTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyObservations_No_Rows()
        {
            var observations = new F64Matrix(0, 10);
            Checks.VerifyObservations(observations);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyObservations_No_Columns()
        {
            var observations = new F64Matrix(10, 0);
            Checks.VerifyObservations(observations);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_Verify_Targets()
        {
            var targets = new double[0];
            Checks.VerifyTargets(targets);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyObservationsAndTargetsDimensionMatch()
        {
            var observations = new F64Matrix(112, 10);
            var targets = new double[100];
            Checks.VerifyObservationsAndTargetsDimensionMatch(observations, targets);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyIndices_Negative_Values()
        {
            var indices = new int[] { 1, 3, 5, -10 };
            var observations = new F64Matrix(112, 10);
            var targets = new double[100];
            Checks.VerifyIndices(indices, observations, targets);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyIndices_Max_Larger_Than_RowCount()
        {
            var indices = new int[] { 1, 3, 5, 115 };
            var observations = new F64Matrix(100, 10);
            var targets = new double[100];
            Checks.VerifyIndices(indices, observations, targets);
        }
    }
}
