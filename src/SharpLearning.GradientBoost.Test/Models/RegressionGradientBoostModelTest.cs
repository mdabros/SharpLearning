﻿using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.GradientBoost.Loss;
using SharpLearning.GradientBoost.Models;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.GradientBoost.Test.Models
{
    [TestClass]
    public class RegressionGradientBoostModelTest
    {
        readonly string m_regressionGradientBoostModelString = "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<RegressionGradientBoostModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.GradientBoost.Models\">\r\n  <FeatureCount>2</FeatureCount>\r\n  <InitialLoss>0.38461538461538464</InitialLoss>\r\n  <LearningRate>0.1</LearningRate>\r\n  <Trees xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.GradientBoost.GBMDecisionTree\" z:Id=\"2\" z:Size=\"2\">\r\n    <d2p1:GBMTree z:Id=\"3\">\r\n      <d2p1:Nodes z:Id=\"4\" z:Size=\"5\">\r\n        <d2p1:GBMNode z:Id=\"5\">\r\n          <d2p1:Depth>0</d2p1:Depth>\r\n          <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>1.2810265668751807E-17</d2p1:LeftConstant>\r\n          <d2p1:LeftError>6.153846153846156</d2p1:LeftError>\r\n          <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>1.2810265668751807E-17</d2p1:RightConstant>\r\n          <d2p1:RightError>6.153846153846156</d2p1:RightError>\r\n          <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n          <d2p1:SampleCount>26</d2p1:SampleCount>\r\n          <d2p1:SplitValue>-1</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n        <d2p1:GBMNode z:Id=\"6\">\r\n          <d2p1:Depth>1</d2p1:Depth>\r\n          <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>-0.11188811188811187</d2p1:LeftConstant>\r\n          <d2p1:LeftError>4.3636363636363651</d2p1:LeftError>\r\n          <d2p1:LeftIndex>2</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>0.61538461538461542</d2p1:RightConstant>\r\n          <d2p1:RightError>-4.4408920985006262E-16</d2p1:RightError>\r\n          <d2p1:RightIndex>3</d2p1:RightIndex>\r\n          <d2p1:SampleCount>26</d2p1:SampleCount>\r\n          <d2p1:SplitValue>20</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n        <d2p1:GBMNode z:Id=\"7\">\r\n          <d2p1:Depth>2</d2p1:Depth>\r\n          <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>0.61538461538461542</d2p1:LeftConstant>\r\n          <d2p1:LeftError>0</d2p1:LeftError>\r\n          <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>-0.1465201465201465</d2p1:RightConstant>\r\n          <d2p1:RightError>3.8095238095238111</d2p1:RightError>\r\n          <d2p1:RightIndex>4</d2p1:RightIndex>\r\n          <d2p1:SampleCount>22</d2p1:SampleCount>\r\n          <d2p1:SplitValue>2.5</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n        <d2p1:GBMNode z:Id=\"8\">\r\n          <d2p1:Depth>2</d2p1:Depth>\r\n          <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>0.61538461538461542</d2p1:LeftConstant>\r\n          <d2p1:LeftError>-2.2204460492503131E-16</d2p1:LeftError>\r\n          <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>0.61538461538461542</d2p1:RightConstant>\r\n          <d2p1:RightError>-5.5511151231257827E-16</d2p1:RightError>\r\n          <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n          <d2p1:SampleCount>4</d2p1:SampleCount>\r\n          <d2p1:SplitValue>25</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n        <d2p1:GBMNode z:Id=\"9\">\r\n          <d2p1:Depth>3</d2p1:Depth>\r\n          <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>-0.25128205128205128</d2p1:LeftConstant>\r\n          <d2p1:LeftError>1.7333333333333338</d2p1:LeftError>\r\n          <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>0.11538461538461538</d2p1:RightConstant>\r\n          <d2p1:RightError>1.4999999999999998</d2p1:RightError>\r\n          <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n          <d2p1:SampleCount>21</d2p1:SampleCount>\r\n          <d2p1:SplitValue>13.5</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n      </d2p1:Nodes>\r\n    </d2p1:GBMTree>\r\n    <d2p1:GBMTree z:Id=\"10\">\r\n      <d2p1:Nodes z:Id=\"11\" z:Size=\"6\">\r\n        <d2p1:GBMNode z:Id=\"12\">\r\n          <d2p1:Depth>0</d2p1:Depth>\r\n          <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>-1.2810265668751807E-17</d2p1:LeftConstant>\r\n          <d2p1:LeftError>5.5989487179487192</d2p1:LeftError>\r\n          <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>-1.2810265668751807E-17</d2p1:RightConstant>\r\n          <d2p1:RightError>5.5989487179487192</d2p1:RightError>\r\n          <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n          <d2p1:SampleCount>26</d2p1:SampleCount>\r\n          <d2p1:SplitValue>-1</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n        <d2p1:GBMNode z:Id=\"13\">\r\n          <d2p1:Depth>1</d2p1:Depth>\r\n          <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>-0.10069930069930067</d2p1:LeftConstant>\r\n          <d2p1:LeftError>4.14887878787879</d2p1:LeftError>\r\n          <d2p1:LeftIndex>2</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>0.55384615384615365</d2p1:RightConstant>\r\n          <d2p1:RightError>-1.1102230246251565E-15</d2p1:RightError>\r\n          <d2p1:RightIndex>3</d2p1:RightIndex>\r\n          <d2p1:SampleCount>26</d2p1:SampleCount>\r\n          <d2p1:SplitValue>20</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n        <d2p1:GBMNode z:Id=\"14\">\r\n          <d2p1:Depth>2</d2p1:Depth>\r\n          <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>0.27829059829059832</d2p1:LeftConstant>\r\n          <d2p1:LeftError>0.61389629629629627</d2p1:LeftError>\r\n          <d2p1:LeftIndex>4</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>-0.1605398110661268</d2p1:RightConstant>\r\n          <d2p1:RightError>3.0360456140350895</d2p1:RightError>\r\n          <d2p1:RightIndex>5</d2p1:RightIndex>\r\n          <d2p1:SampleCount>22</d2p1:SampleCount>\r\n          <d2p1:SplitValue>4.5</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n        <d2p1:GBMNode z:Id=\"15\">\r\n          <d2p1:Depth>2</d2p1:Depth>\r\n          <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>0.55384615384615388</d2p1:LeftConstant>\r\n          <d2p1:LeftError>0</d2p1:LeftError>\r\n          <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>0.55384615384615354</d2p1:RightConstant>\r\n          <d2p1:RightError>-1.2212453270876722E-15</d2p1:RightError>\r\n          <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n          <d2p1:SampleCount>4</d2p1:SampleCount>\r\n          <d2p1:SplitValue>23</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n        <d2p1:GBMNode z:Id=\"16\">\r\n          <d2p1:Depth>3</d2p1:Depth>\r\n          <d2p1:FeatureIndex>0</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>-0.35948717948717951</d2p1:LeftConstant>\r\n          <d2p1:LeftError>0</d2p1:LeftError>\r\n          <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>0.59717948717948721</d2p1:RightConstant>\r\n          <d2p1:RightError>0.003755555555555623</d2p1:RightError>\r\n          <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n          <d2p1:SampleCount>3</d2p1:SampleCount>\r\n          <d2p1:SplitValue>4</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n        <d2p1:GBMNode z:Id=\"17\">\r\n          <d2p1:Depth>3</d2p1:Depth>\r\n          <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n          <d2p1:LeftConstant>-0.28256410256410258</d2p1:LeftConstant>\r\n          <d2p1:LeftError>0.92307692307692291</d2p1:LeftError>\r\n          <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n          <d2p1:RightConstant>0.10384615384615396</d2p1:RightConstant>\r\n          <d2p1:RightError>1.4999999999999998</d2p1:RightError>\r\n          <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n          <d2p1:SampleCount>19</d2p1:SampleCount>\r\n          <d2p1:SplitValue>13.5</d2p1:SplitValue>\r\n        </d2p1:GBMNode>\r\n      </d2p1:Nodes>\r\n    </d2p1:GBMTree>\r\n  </Trees>\r\n</RegressionGradientBoostModel>";

        [TestMethod]
        public void RegressionGradientBoostModel_Predict_Single()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learner = new RegressionGradientBoostLearner(100, 0.1, 3, 1, 1e-6, 1.0, 0, 
                new GradientBoostSquaredLoss(), false);

            var sut = learner.Learn(observations, targets);

            var rows = targets.Length;
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.045093177702025665, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionGradientBoostModel_Precit_Multiple()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learner = new RegressionGradientBoostLearner(100, 0.1, 3, 1, 1e-6, 1.0, 0, 
                new GradientBoostSquaredLoss(), false);

            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.045093177702025665, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionGradientBoostModel_GetVariableImportance()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new RegressionGradientBoostLearner(100, 0.1, 3, 1, 1e-6, 1.0, 0, 
                new GradientBoostSquaredLoss(), false);

            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 72.1682473281495 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void RegressionGradientBoostModel_GetRawVariableImportance()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learner = new RegressionGradientBoostLearner(100, 0.1, 3, 1, 1e-6, 1.0, 0, 
                new GradientBoostSquaredLoss(), false);

            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 31.124562320186836, 43.127779144563753 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        public void RegressionGradientBoostModel_Save()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learner = new RegressionGradientBoostLearner(2, 0.1, 3, 1, 1e-6, 1.0, 0, 
                new GradientBoostSquaredLoss(), false);

            var sut = learner.Learn(observations, targets);

            // save model.
            var writer = new StringWriter();
            sut.Save(() => writer);

            // load model and assert prediction results.
            sut = RegressionGradientBoostModel.Load(() => new StringReader(writer.ToString()));
            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.192163332018409, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionGradientBoostModel_Load()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var reader = new StringReader(m_regressionGradientBoostModelString);
            var sut = RegressionGradientBoostModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.42445562130177528, error, 0.0000001);
        }
    }
}
