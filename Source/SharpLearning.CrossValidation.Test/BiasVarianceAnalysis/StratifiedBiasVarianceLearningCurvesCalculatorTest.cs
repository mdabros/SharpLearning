using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.BiasVarianceAnalysis;
using SharpLearning.CrossValidation.Test.Properties;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System.Collections.Generic;
using System.IO;

namespace SharpLearning.CrossValidation.Test.BiasVarianceAnalysis
{
    [TestClass]
    public class StratifiedBiasVarianceLearningCurvesCalculatorTest
    {
        [TestMethod]
        public void StratifiedBiasVarianceLearningCurvesCalculator_Calculate()
        {
            var sut = new StratifiedBiasVarianceLearningCurvesCalculator<double>(new TotalErrorClassificationMetric<double>(),
                new double[] { 0.2, 0.8 }, 0.8, 5, 42);

            var targetName = "Pass";
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => !v.Contains(targetName)).ToF64Matrix();
            var targets = parser.EnumerateRows(targetName).ToF64Vector();

            var actual = sut.Calculate(new ClassificationDecisionTreeLearner(),
                observations, targets);

            var expected = new List<BiasVarianceLearningCurvePoint>() { new BiasVarianceLearningCurvePoint(4, 0, 0.39999999999999997), 
                new BiasVarianceLearningCurvePoint(16, 0.0625, 0.33333333333333331)};

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
