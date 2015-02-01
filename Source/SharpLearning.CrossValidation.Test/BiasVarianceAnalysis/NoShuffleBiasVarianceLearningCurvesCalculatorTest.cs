using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.BiasVarianceAnalysis;
using SharpLearning.CrossValidation.Test.Properties;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.InputOutput.Csv;
using System.Collections.Generic;
using System.IO;

namespace SharpLearning.CrossValidation.Test.BiasVarianceAnalysis
{
    [TestClass]
    public class NoShuffleBiasVarianceLearningCurvesCalculatorTest
    {
        [TestMethod]
        public void NoShuffleBiasVarianceLearningCurvesCalculator_Calculate()
        {
            var sut = new NoShuffleBiasVarianceLearningCurvesCalculator<double>(new CrossValidationTestMetric(), 
                new double[] { 0.2, 0.8 }, 0.8 );

            var targetName = "T";
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows(v => !v.Contains(targetName)).ToF64Matrix();
            var targets = parser.EnumerateRows(targetName).ToF64Vector();

            var actual = sut.Calculate(() => new RegressionDecisionTreeLearner(),
                observations, targets);

            var expected = new List<BiasVarianceLearningCurvePoint>() { new BiasVarianceLearningCurvePoint(32, 0, 0.12874833873980004), 
                new BiasVarianceLearningCurvePoint(128, 0.0, 0.067720786718774989)};
            
            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
