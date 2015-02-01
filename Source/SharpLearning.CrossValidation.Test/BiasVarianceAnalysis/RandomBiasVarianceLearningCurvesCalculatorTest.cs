﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.BiasVarianceAnalysis;
using SharpLearning.CrossValidation.Test.Properties;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.InputOutput.Csv;
using System.Collections.Generic;
using System.IO;

namespace SharpLearning.CrossValidation.Test.BiasVarianceAnalysis
{
    [TestClass]
    public class RandomBiasVarianceLearningCurvesCalculatorTest
    {
        [TestMethod]
        public void RandomBiasVarianceLearningCurvesCalculator_Calculate()
        {
            var sut = new RandomBiasVarianceLearningCurvesCalculator<double>(new CrossValidationTestMetric(), 
                new double[] { 0.2, 0.8 }, 0.8, 42);

            var targetName = "T";
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows(v => !v.Contains(targetName)).ToF64Matrix();
            var targets = parser.EnumerateRows(targetName).ToF64Vector();

            var actual = sut.Calculate(() => new RegressionDecisionTreeLearner(),
                observations, targets);

            var expected = new List<BiasVarianceLearningCurvePoint>() { new BiasVarianceLearningCurvePoint(32, 0, 0.22543776658935008), 
                new BiasVarianceLearningCurvePoint(128, 0.0, 0.0656601097562)};

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}