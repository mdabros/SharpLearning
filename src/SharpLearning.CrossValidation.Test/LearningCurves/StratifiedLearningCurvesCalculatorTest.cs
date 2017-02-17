using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.LearningCurves;
using SharpLearning.CrossValidation.Test.Properties;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System.Collections.Generic;
using System.IO;

namespace SharpLearning.CrossValidation.Test.LearningCurves
{
    [TestClass]
    public class StratifiedLearningCurvesCalculatorTest
    {
        [TestMethod]
        public void StratifiedLearningCurvesCalculator_Calculate()
        {
            var sut = new StratifiedLearningCurvesCalculator<double>(new TotalErrorClassificationMetric<double>(),
                new double[] { 0.2, 0.8 }, 0.8, 5, 42);

            var targetName = "Pass";
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => !v.Contains(targetName)).ToF64Matrix();
            var targets = parser.EnumerateRows(targetName).ToF64Vector();

            var actual = sut.Calculate(new ClassificationDecisionTreeLearner(),
                observations, targets);

            var expected = new List<LearningCurvePoint>() { new LearningCurvePoint(4, 0, 0.39999999999999997), 
                new LearningCurvePoint(16, 0.0625, 0.33333333333333331)};

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
