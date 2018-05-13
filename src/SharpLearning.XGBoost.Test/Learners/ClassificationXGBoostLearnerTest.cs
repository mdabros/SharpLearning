﻿using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.XGBoost.Learners;
using SharpLearning.XGBoost.Test.Properties;

namespace SharpLearning.XGBoost.Test.Learners
{
    [TestClass]
    public class ClassificationXGBoostLearnerTest
    {
        readonly double m_delta = 0.0000001;

        [TestMethod]
        public void ClassificationXGBoostLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = CreateLearner();

            using (var model = sut.Learn(observations, targets))
            {
                var predictions = model.Predict(observations);

                var evaluator = new TotalErrorClassificationMetric<double>();
                var error = evaluator.Error(targets, predictions);

                Assert.AreEqual(0.17757009345794392, error, m_delta);
            }
        }

        [TestMethod]
        public void ClassificationXGBoostLearner_Learn_indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var sut = CreateLearner();

            using (var model = sut.Learn(observations, targets, indices))
            {
                var predictions = model.Predict(observations);

                var evaluator = new TotalErrorClassificationMetric<double>();
                var error = evaluator.Error(targets, predictions);

                Assert.AreEqual(0.228971962616822, error, m_delta);
            }
        }

        static ClassificationXGBoostLearner CreateLearner()
        {
            return new ClassificationXGBoostLearner(maximumTreeDepth: 3,
                learningRate: 0.1,
                estimators: 2,
                silent: true,
                objective: ClassificationObjective.Softmax,
                boosterType: BoosterType.GBTree,
                treeMethod: TreeMethod.Auto,
                numberOfThreads: -1,
                gamma: 0,
                minChildWeight: 1,
                maxDeltaStep: 0,
                subSample: 1,
                colSampleByTree: 1,
                colSampleByLevel: 1,
                l1Regularization: 0,
                l2Reguralization: 1,
                scalePosWeight: 1,
                baseScore: 0.5,
                seed: 0,
                missing: double.NaN);
        }
    }
}
