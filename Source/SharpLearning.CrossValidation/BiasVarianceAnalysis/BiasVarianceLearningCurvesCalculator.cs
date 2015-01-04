using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.CrossValidation.BiasVarianceAnalysis
{
    /// <summary>
    /// Bias variance analysis calculator for constructing learning curves.
    /// Learning curves can be used to determine if a model has high bias or high variance.
    /// 
    /// Solutions for model with high bias:
    ///  - Add more features.
    ///  - Use a more sophisticated model
    ///  - Decrease regularization.
    /// Solutions for model with high variance
    ///  - Use fewer features.
    ///  - Use more training samples.
    ///  - Increase Regularization.
    /// </summary>
    public sealed class BiasVarianceLearningCurvesCalculator<TPrediction>
    {
        readonly double[] m_samplePercentages;
        readonly IMetric<double, TPrediction> m_metric;
        readonly double m_trainingPercentage;
        readonly Random m_random;

        /// <summary>
        /// Bias variance analysis calculator for constructing learning curves.
        /// Learning curves can be used to determine if a model has high bias or high variance.
        /// </summary>
        /// <param name="metric">The error metric used</param>
        /// <param name="samplePercentages">A list of sample percentages determining the 
        /// training data used in each point of the learning curve</param>
        /// <param name="trainingPercentage">The percentage of the data used for training. 
        /// the sample percentage list is sampled from this set while the remaining data is used for validation</param>
        /// <param name="seed"></param>
        public BiasVarianceLearningCurvesCalculator(IMetric<double, TPrediction> metric, double[] samplePercentages, 
            double trainingPercentage = 0.8, int seed = 42)
        {
            if (samplePercentages == null) { throw new ArgumentNullException("samplePercentages"); }
            if (samplePercentages.Length < 1) { throw new ArgumentException("SamplePercentages length must be at least 1"); }
            if (metric == null) { throw new ArgumentNullException("metric");}
            if (trainingPercentage <= 0.0 || trainingPercentage >= 1.0) 
            { throw new ArgumentException("Training percentage must be larger than 0.0 and smaller than 1.0"); }
            
            m_samplePercentages = samplePercentages;
            m_metric = metric;
            m_trainingPercentage = trainingPercentage; 
            m_random = new Random(seed);
        }

        /// <summary>
        /// Returns a list of BiasVarianceLearningCurvePoints for constructing learning curves.
        /// The points contain sample size, training score and validation score. 
        /// </summary>
        /// <param name="learnerFactory"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public List<BiasVarianceLearningCurvePoint> Calculate(Func<IIndexedLearner<TPrediction>> learnerFactory,
            F64Matrix observations, double[] targets)
        {
            var learningCurves = new List<BiasVarianceLearningCurvePoint>();
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(m_random);

            var trainingSampleSize = (int)(m_trainingPercentage * (double)indices.Length);
            trainingSampleSize = trainingSampleSize > 0 ? trainingSampleSize : 1;

            var trainingIndices = indices.Take(trainingSampleSize)
                .ToArray();
            var validationIndices = indices.Except(trainingIndices)
                .ToArray();

            var validationTargets = targets.GetIndices(validationIndices);

            foreach (var samplePercentage in m_samplePercentages)
            {
                if (samplePercentage <= 0.0 || samplePercentage > 1.0)
                { throw new ArgumentException("Sample percentage must be larger than 0.0 and smaller than or equal to 1.0"); }

                var sampleSize = (int)(samplePercentage * trainingIndices.Length);
                sampleSize = sampleSize > 0 ? sampleSize : 1;

                var sampleIndices = trainingIndices.Take(sampleSize).ToArray();

                var model = learnerFactory().Learn(observations, targets, sampleIndices);
                var trainingPredictions = new TPrediction[sampleSize];
                
                for (int i = 0; i < trainingPredictions.Length; i++)
                {
                    trainingPredictions[i] = model.Predict(observations.GetRow(sampleIndices[i]));
                }

                var validationPredictions = new TPrediction[validationIndices.Length];
                for (int i = 0; i < validationIndices.Length; i++)
                {
                    validationPredictions[i] = model.Predict(observations.GetRow(validationIndices[i]));
                }

                var trainingTargets = targets.GetIndices(sampleIndices);
                learningCurves.Add(new BiasVarianceLearningCurvePoint(sampleSize,
                    m_metric.Error(trainingTargets, trainingPredictions),
                    m_metric.Error(validationTargets, validationPredictions)));
            }

            return learningCurves;
        }
    }
}
