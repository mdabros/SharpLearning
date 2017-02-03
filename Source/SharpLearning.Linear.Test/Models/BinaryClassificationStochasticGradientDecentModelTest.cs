using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Linear.Learners;
using SharpLearning.Linear.Models;
using SharpLearning.Linear.Test.Properties;
using SharpLearning.Metrics.Classification;
using SharpLearning.Metrics.Regression;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.Linear.Test.Models
{
    [TestClass]
    public class BinaryClassificationStochasticGradientDecentModelTest
    {
        BinaryClassificationStochasticGradientDecentModel m_sut =
            new BinaryClassificationStochasticGradientDecentModel(
                new double[] { -15.934288283579136, 0.14011316001536858, 0.13571128043372779 });

        [TestMethod]
        public void BinaryClassificationStochasticGradientDecentModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var predictions = new double[targets.Length];
            for (int i = 0; i < predictions.Length; i++)
            {
                predictions[i] = m_sut.Predict(observations.Row(i));
            }

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);
            Assert.AreEqual(0.08, actual, 0.001);
        }

        [TestMethod]
        public void BinaryClassificationStochasticGradientDecentModel_Predict_Multi()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var metric = new TotalErrorClassificationMetric<double>();

            var predictions = m_sut.Predict(observations);
            var actual = metric.Error(targets, predictions);
            Assert.AreEqual(0.08, actual, 0.001);
        }

        [TestMethod]
        public void BinaryClassificationStochasticGradientDecentModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var actual = m_sut.GetRawVariableImportance();
            Assert.AreEqual(0.14011316001536858, actual[0], 0.001);
            Assert.AreEqual(0.13571128043372779, actual[1], 0.001);
        }

        [TestMethod]
        public void BinaryClassificationStochasticGradientDecentModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var featureNameToIndex = parser.EnumerateRows("F1", "F2").First().ColumnNameToIndex;
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var actual = m_sut.GetVariableImportance(featureNameToIndex).ToList();
            var expected = new Dictionary<string, double> { { "F1", 100.0 }, { "F2", 0.879318595984989 } }.ToList();
            
            Assert.AreEqual(expected.Count, actual.Count);
            for (int i = 0; i < expected.Count; i++)
            {
                Assert.AreEqual(expected[i].Key, actual[i].Key);
                Assert.AreEqual(expected[i].Value, actual[i].Value, 0.0001);
            }
        }

        [TestMethod]
        public void BinaryClassificationStochasticGradientDecentModel_PredictProbability_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var featureNameToIndex = parser.EnumerateRows("F1", "F2").First().ColumnNameToIndex;
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = m_sut.PredictProbability(observations.Row(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.08, error, 0.0000001);
            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.621101688132576 }, { 1, 0.378898311867424 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.996775267107844 }, { 1, 0.00322473289215573 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.734571906910202 }, { 1, 0.265428093089798 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0146128977972946 }, { 1, 0.985387102202705 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0046588360759221 }, { 1, 0.995341163924078 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.878121560867105 }, { 1, 0.121878439132895 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00325193200227081 }, { 1, 0.996748067997729 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.289961572914446 }, { 1, 0.710038427085554 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00136926544946114 }, { 1, 0.998630734550539 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.141385058527512 }, { 1, 0.858614941472488 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0638775362194205 }, { 1, 0.93612246378058 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.780859048908631 }, { 1, 0.219140951091369 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00252880042425474 }, { 1, 0.997471199575745 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000870012794030317 }, { 1, 0.99912998720597 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.518839497800346 }, { 1, 0.481160502199654 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0233345267349946 }, { 1, 0.976665473265005 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.288881083312414 }, { 1, 0.711118916687586 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.519797291829792 }, { 1, 0.480202708170208 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00138927920018173 }, { 1, 0.998610720799818 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.212813784372327 }, { 1, 0.787186215627673 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.663944289044127 }, { 1, 0.336055710955873 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0038327815196646 }, { 1, 0.996167218480335 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.902332538939807 }, { 1, 0.0976674610601925 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.994182189991354 }, { 1, 0.00581781000864554 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0128151441950428 }, { 1, 0.987184855804957 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0924308234966024 }, { 1, 0.907569176503398 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.200333386383767 }, { 1, 0.799666613616233 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0848695208602154 }, { 1, 0.915130479139785 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.610815632228508 }, { 1, 0.389184367771492 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.842803823944142 }, { 1, 0.157196176055859 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0726545903870117 }, { 1, 0.927345409612988 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0224314022017195 }, { 1, 0.977568597798281 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.516529956095136 }, { 1, 0.483470043904864 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.314151760775334 }, { 1, 0.685848239224666 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.654369918912129 }, { 1, 0.345630081087871 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.767308909284226 }, { 1, 0.232691090715774 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0966147800476704 }, { 1, 0.90338521995233 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0173290689348947 }, { 1, 0.982670931065105 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.45347207916175 }, { 1, 0.54652792083825 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.717906972566941 }, { 1, 0.282093027433059 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0303782758955657 }, { 1, 0.969621724104434 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.913160268853319 }, { 1, 0.0868397311466806 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00203157004984977 }, { 1, 0.99796842995015 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.246194313426827 }, { 1, 0.753805686573173 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.928453252401366 }, { 1, 0.071546747598634 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.537630876318062 }, { 1, 0.462369123681938 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0116187961764024 }, { 1, 0.988381203823598 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 7.22906642570686E-05 }, { 1, 0.999927709335743 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00274401349818609 }, { 1, 0.997255986501814 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000132042182498737 }, { 1, 0.999867957817501 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00481206796842226 }, { 1, 0.995187932031578 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00192029221819012 }, { 1, 0.99807970778181 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0665401709392388 }, { 1, 0.933459829060761 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.947870386295566 }, { 1, 0.0521296137044343 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.89373284753283 }, { 1, 0.10626715246717 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.704702607811644 }, { 1, 0.295297392188356 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000830673883768451 }, { 1, 0.999169326116232 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.167422259169371 }, { 1, 0.832577740830629 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0189854860794952 }, { 1, 0.981014513920505 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00838929356238649 }, { 1, 0.991610706437614 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00189201342716983 }, { 1, 0.99810798657283 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.990136730682396 }, { 1, 0.00986326931760379 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.938437506110058 }, { 1, 0.0615624938899419 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.993255936266246 }, { 1, 0.00674406373375404 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.658732454672987 }, { 1, 0.341267545327013 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.737220117135673 }, { 1, 0.262779882864327 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0487546269060146 }, { 1, 0.951245373093985 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.882615964636338 }, { 1, 0.117384035363662 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000403421522657044 }, { 1, 0.999596578477343 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.155116073518434 }, { 1, 0.844883926481566 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.995815258541467 }, { 1, 0.00418474145853271 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0258754743699977 }, { 1, 0.974124525630002 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000701507303933036 }, { 1, 0.999298492696067 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0789664837731302 }, { 1, 0.92103351622687 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0688204496211201 }, { 1, 0.93117955037888 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000380276277832881 }, { 1, 0.999619723722167 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0632553662563583 }, { 1, 0.936744633743642 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.194063889277119 }, { 1, 0.805936110722881 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.844854177593534 }, { 1, 0.155145822406466 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.20031632135846 }, { 1, 0.79968367864154 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00248032992297498 }, { 1, 0.997519670077025 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0278005002001657 }, { 1, 0.972199499799834 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0734439511455673 }, { 1, 0.926556048854433 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.45949715632221 }, { 1, 0.54050284367779 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000466181138822686 }, { 1, 0.999533818861177 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00509105733882598 }, { 1, 0.994908942661174 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.340478301551184 }, { 1, 0.659521698448816 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00099401770743679 }, { 1, 0.999005982292563 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000274656235010107 }, { 1, 0.99972534376499 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.587232575620937 }, { 1, 0.412767424379063 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00044325289131486 }, { 1, 0.999556747108685 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000181391006671028 }, { 1, 0.999818608993329 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.965549093826312 }, { 1, 0.0344509061736877 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00242968647505071 }, { 1, 0.997570313524949 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0567903397133381 }, { 1, 0.943209660286662 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0887177518312935 }, { 1, 0.911282248168707 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.140867030502927 }, { 1, 0.859132969497073 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000665271179681626 }, { 1, 0.999334728820318 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.347153973096688 }, { 1, 0.652846026903312 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00123801906031829 }, { 1, 0.998761980939682 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void BinaryClassificationStochasticGradientDecentModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var featureNameToIndex = parser.EnumerateRows("F1", "F2").First().ColumnNameToIndex;
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var actual = m_sut.PredictProbability(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.08, error, 0.0000001);
            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.621101688132576 }, { 1, 0.378898311867424 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.996775267107844 }, { 1, 0.00322473289215573 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.734571906910202 }, { 1, 0.265428093089798 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0146128977972946 }, { 1, 0.985387102202705 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0046588360759221 }, { 1, 0.995341163924078 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.878121560867105 }, { 1, 0.121878439132895 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00325193200227081 }, { 1, 0.996748067997729 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.289961572914446 }, { 1, 0.710038427085554 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00136926544946114 }, { 1, 0.998630734550539 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.141385058527512 }, { 1, 0.858614941472488 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0638775362194205 }, { 1, 0.93612246378058 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.780859048908631 }, { 1, 0.219140951091369 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00252880042425474 }, { 1, 0.997471199575745 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000870012794030317 }, { 1, 0.99912998720597 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.518839497800346 }, { 1, 0.481160502199654 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0233345267349946 }, { 1, 0.976665473265005 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.288881083312414 }, { 1, 0.711118916687586 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.519797291829792 }, { 1, 0.480202708170208 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00138927920018173 }, { 1, 0.998610720799818 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.212813784372327 }, { 1, 0.787186215627673 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.663944289044127 }, { 1, 0.336055710955873 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0038327815196646 }, { 1, 0.996167218480335 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.902332538939807 }, { 1, 0.0976674610601925 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.994182189991354 }, { 1, 0.00581781000864554 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0128151441950428 }, { 1, 0.987184855804957 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0924308234966024 }, { 1, 0.907569176503398 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.200333386383767 }, { 1, 0.799666613616233 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0848695208602154 }, { 1, 0.915130479139785 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.610815632228508 }, { 1, 0.389184367771492 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.842803823944142 }, { 1, 0.157196176055859 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0726545903870117 }, { 1, 0.927345409612988 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0224314022017195 }, { 1, 0.977568597798281 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.516529956095136 }, { 1, 0.483470043904864 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.314151760775334 }, { 1, 0.685848239224666 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.654369918912129 }, { 1, 0.345630081087871 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.767308909284226 }, { 1, 0.232691090715774 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0966147800476704 }, { 1, 0.90338521995233 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0173290689348947 }, { 1, 0.982670931065105 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.45347207916175 }, { 1, 0.54652792083825 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.717906972566941 }, { 1, 0.282093027433059 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0303782758955657 }, { 1, 0.969621724104434 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.913160268853319 }, { 1, 0.0868397311466806 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00203157004984977 }, { 1, 0.99796842995015 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.246194313426827 }, { 1, 0.753805686573173 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.928453252401366 }, { 1, 0.071546747598634 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.537630876318062 }, { 1, 0.462369123681938 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0116187961764024 }, { 1, 0.988381203823598 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 7.22906642570686E-05 }, { 1, 0.999927709335743 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00274401349818609 }, { 1, 0.997255986501814 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000132042182498737 }, { 1, 0.999867957817501 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00481206796842226 }, { 1, 0.995187932031578 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00192029221819012 }, { 1, 0.99807970778181 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0665401709392388 }, { 1, 0.933459829060761 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.947870386295566 }, { 1, 0.0521296137044343 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.89373284753283 }, { 1, 0.10626715246717 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.704702607811644 }, { 1, 0.295297392188356 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000830673883768451 }, { 1, 0.999169326116232 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.167422259169371 }, { 1, 0.832577740830629 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0189854860794952 }, { 1, 0.981014513920505 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00838929356238649 }, { 1, 0.991610706437614 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00189201342716983 }, { 1, 0.99810798657283 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.990136730682396 }, { 1, 0.00986326931760379 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.938437506110058 }, { 1, 0.0615624938899419 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.993255936266246 }, { 1, 0.00674406373375404 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.658732454672987 }, { 1, 0.341267545327013 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.737220117135673 }, { 1, 0.262779882864327 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0487546269060146 }, { 1, 0.951245373093985 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.882615964636338 }, { 1, 0.117384035363662 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000403421522657044 }, { 1, 0.999596578477343 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.155116073518434 }, { 1, 0.844883926481566 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.995815258541467 }, { 1, 0.00418474145853271 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0258754743699977 }, { 1, 0.974124525630002 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000701507303933036 }, { 1, 0.999298492696067 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0789664837731302 }, { 1, 0.92103351622687 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0688204496211201 }, { 1, 0.93117955037888 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000380276277832881 }, { 1, 0.999619723722167 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0632553662563583 }, { 1, 0.936744633743642 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.194063889277119 }, { 1, 0.805936110722881 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.844854177593534 }, { 1, 0.155145822406466 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.20031632135846 }, { 1, 0.79968367864154 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00248032992297498 }, { 1, 0.997519670077025 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0278005002001657 }, { 1, 0.972199499799834 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0734439511455673 }, { 1, 0.926556048854433 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.45949715632221 }, { 1, 0.54050284367779 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000466181138822686 }, { 1, 0.999533818861177 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00509105733882598 }, { 1, 0.994908942661174 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.340478301551184 }, { 1, 0.659521698448816 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00099401770743679 }, { 1, 0.999005982292563 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000274656235010107 }, { 1, 0.99972534376499 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.587232575620937 }, { 1, 0.412767424379063 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00044325289131486 }, { 1, 0.999556747108685 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000181391006671028 }, { 1, 0.999818608993329 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.965549093826312 }, { 1, 0.0344509061736877 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00242968647505071 }, { 1, 0.997570313524949 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0567903397133381 }, { 1, 0.943209660286662 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0887177518312935 }, { 1, 0.911282248168707 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.140867030502927 }, { 1, 0.859132969497073 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.000665271179681626 }, { 1, 0.999334728820318 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.347153973096688 }, { 1, 0.652846026903312 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.00123801906031829 }, { 1, 0.998761980939682 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void BinaryClassificationStochasticGradientDecentModel_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var writer = new StringWriter();
            m_sut.Save(() => writer);

            Assert.AreEqual(BinaryClassificationStochasticGradientDecentModelString, writer.ToString());
        }

        [TestMethod]
        public void BinaryClassificationStochasticGradientDecentModel_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var reader = new StringReader(BinaryClassificationStochasticGradientDecentModelString);
            var sut = BinaryClassificationStochasticGradientDecentModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);
            Assert.AreEqual(0.08, actual, 0.001);
        }

        readonly string BinaryClassificationStochasticGradientDecentModelString =
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<BinaryClassificationStochasticGradientDecentModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Linear.Models\">\r\n  <m_weights xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"2\" z:Size=\"3\">\r\n    <d2p1:double>-15.934288283579136</d2p1:double>\r\n    <d2p1:double>0.14011316001536858</d2p1:double>\r\n    <d2p1:double>0.13571128043372779</d2p1:double>\r\n  </m_weights>\r\n</BinaryClassificationStochasticGradientDecentModel>";
    }
}
