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
        public void Checks_VerifyAllLearnerInputs_Observations_Rows_Not_Valid()
        {
            var rowCount = 0;
            var columnCount = 10;
            var targetLength = 10;
            var indices = new int[] { 1, 3, 5, 6 };

            Checks.VerifyAllLearnerInputs(rowCount, columnCount, targetLength, indices);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyAllLearnerInputs_Observations_Cols_Not_Valid()
        {
            var rowCount = 10;
            var columnCount = 0;
            var targetLength = 10;
            var indices = new int[] { 1, 3, 5, 6 };

            Checks.VerifyAllLearnerInputs(rowCount, columnCount, targetLength, indices);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyAllLearnerInputs_Target_Length_Not_Valid()
        {
            var rowCount = 10;
            var columnCount = 5;
            var targetLength = 0;
            var indices = new int[] { 1, 3, 5, 6 };

            Checks.VerifyAllLearnerInputs(rowCount, columnCount, targetLength, indices);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyAllLearnerInputs_ObservationRows_And_Target_Length_Not_Valid()
        {
            var rowCount = 10;
            var columnCount = 5;
            var targetLength = 100;
            var indices = new int[] { 1, 3, 5, 6 };

            Checks.VerifyAllLearnerInputs(rowCount, columnCount, targetLength, indices);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyAllLearnerInputs_Indices_Contains_Negative_Values()
        {
            var rowCount = 10;
            var columnCount = 5;
            var targetLength = 10;
            var indices = new int[] { 1, 3, 5, -8, 6, -10 };

            Checks.VerifyAllLearnerInputs(rowCount, columnCount, targetLength, indices);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyAllLearnerInputs_Indices_Max_Exceeds_RowCount_And_Targets()
        {
            var rowCount = 10;
            var columnCount = 5;
            var targetLength = 10;
            var indices = new int[] { 1, 3, 5, 6, 100 };

            Checks.VerifyAllLearnerInputs(rowCount, columnCount, targetLength, indices);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyObservations_No_Rows()
        {
            var rowCount = 0;
            var columnCount = 10;
            Checks.VerifyObservations(rowCount, columnCount);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyObservations_No_Columns()
        {
            var rowCount = 10;
            var columnCount = 0;
            Checks.VerifyObservations(rowCount, columnCount);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_Verify_Targets()
        {
            var targetLength = 0;
            Checks.VerifyTargets(targetLength);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyObservationsAndTargetsDimensionMatch()
        {
            var observationRowCount = 112;
            var targetLength = 100;
            Checks.VerifyObservationsRowCountAndTargetsLengthMatch(observationRowCount, targetLength);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyIndices_Negative_Values()
        {
            var indices = new int[] { 1, 3, 5, -10 };
            var observationRowCount = 100;
            var targetLength = 100;
            Checks.VerifyIndices(indices, observationRowCount, targetLength);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Checks_VerifyIndices_Max_Larger_Than_RowCount()
        {
            var indices = new int[] { 1, 3, 5, 115 };
            var observationRowCount = 100;
            var targetLength = 100;
            Checks.VerifyIndices(indices, observationRowCount, targetLength);
        }
    }
}
