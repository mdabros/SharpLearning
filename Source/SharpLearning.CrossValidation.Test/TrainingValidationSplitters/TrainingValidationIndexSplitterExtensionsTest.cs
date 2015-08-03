using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.CrossValidation.TrainingValidationSplitters;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers;
using System.Linq;

namespace SharpLearning.CrossValidation.Test.TrainingValidationSplitters
{
    [TestClass]
    public class TrainingValidationIndexSplitterExtensionsTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TrainingValidationIndexSplitterExtensions_SplitSet_Observations_Targets_Row_Differ()
        {
            var splitter = new RandomTrainingValidationIndexSplitter<double>(0.6, 32);
            splitter.SplitSet(new F64Matrix(10, 2), new double[8]);
        }

        [TestMethod]
        public void TrainingValidationIndexSplitterExtensions_SplitSet()
        {
            var observations = new F64Matrix(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 10, 1);
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

            var splitter = new NoShuffleTrainingValidationIndexSplitter<double>(0.6);
            
            var actual = splitter.SplitSet(observations, targets);

            var trainingIndices = Enumerable.Range(0, 6).ToArray();
            var testIndices = Enumerable.Range(6, 4).ToArray();

            var expected = new TrainingTestSetSplit(
                new ObservationTargetSet((F64Matrix)observations.GetRows(trainingIndices), 
                    targets.GetIndices(trainingIndices)),
                new ObservationTargetSet((F64Matrix)observations.GetRows(testIndices), 
                    targets.GetIndices(testIndices)));

            Assert.AreEqual(expected, actual);
        }
    }
}
