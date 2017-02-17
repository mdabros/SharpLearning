using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.LearningCurves;
using SharpLearning.CrossValidation.Test.Properties;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System.Collections.Generic;
using System.IO;

namespace SharpLearning.CrossValidation.Test.LearningCurves
{
    [TestClass]
    public class RandomLearningCurvesCalculatorTest
    {
        [TestMethod]
        public void RandomLearningCurvesCalculator_Calculate()
        {
            var sut = new RandomShuffleLearningCurvesCalculator<double>(new MeanSquaredErrorRegressionMetric(), 
                new double[] { 0.2, 0.8 }, 0.8, 42, 5);

            var targetName = "T";
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows(v => !v.Contains(targetName)).ToF64Matrix();
            var targets = parser.EnumerateRows(targetName).ToF64Vector();

            var actual = sut.Calculate(new RegressionDecisionTreeLearner(),
                observations, targets);

            var expected = new List<LearningCurvePoint>() { new LearningCurvePoint(32, 0, 0.141565953928265), 
                new LearningCurvePoint(128, 0.0, 0.068970597423950036)};

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
