using SharpLearning.Containers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Metrics.Classification
{
    /// <summary>
    /// The logarithm of the likelihood function for a Bernoulli random distribution.
    /// In plain English, this error metric is typically used where you have to predict that something is true or false 
    /// with a probability (likelihood) ranging from definitely true (1) to equally true (0.5) to definitely false(0).
    /// The use of log on the error provides extreme punishments for being both confident and wrong. 
    /// In the worst possible case, a single prediction that something is definitely true (1) 
    /// when it is actually false will add infinite to your error score and make every other entry pointless.
    /// https://www.kaggle.com/wiki/MultiClassLogLoss
    /// </summary>
    public sealed class LogLossClassificationProbabilityMetric : IClassificationProbabilityMetric
    {
        readonly double m_epsilon;
        readonly ClassificationMatrixStringConverter<double> m_converter = new ClassificationMatrixStringConverter<double>();
        readonly ClassificationMatrix<double> m_classificationMatrix = new ClassificationMatrix<double>();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="epsilon"></param>
        public LogLossClassificationProbabilityMetric(double epsilon = 1e-15)
        {
            m_epsilon = epsilon;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Error(double[] targets, ProbabilityPrediction[] predictions)
        {
            var rows = targets.Length;
            var sum = 0.0;
            for (int i = 0; i < rows; i++)
            {
                var probabilities = predictions[i].Probabilities;
                var target = targets[i];
                var probabilitySum = probabilities.Select(p => p.Value)
                    .Sum();
                
                foreach (var probability in probabilities)
                {
                    if(probability.Key == target)
                    {
                        var prop = Math.Max(m_epsilon, probability.Value);
                        prop = Math.Min(1.0 - m_epsilon, prop);
                        sum += Math.Log(prop / probabilitySum);
                    }
                }
            }

            return -1.0 / (double)rows * sum;
        }

        List<double> UniqueTargets(double[] targets, double[] predictions)
        {
            var uniquePredictions = predictions.Distinct();
            var uniqueTargets = targets.Distinct();
            var uniques = uniqueTargets.Union(uniquePredictions).ToList();

            uniques.Sort();
            return uniques;
        }

        /// <summary>
        /// Creates an error matrix based on the provided confusion matrix
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public string ErrorString(double[] targets, ProbabilityPrediction[] predictions)
        {
            var classPredictions = predictions.Select(p => p.Prediction).ToArray();
            var uniques = UniqueTargets(targets, classPredictions);

            var confusionMatrix = m_classificationMatrix.ConfusionMatrix(uniques, targets, classPredictions);
            var errorMatrix = m_classificationMatrix.ErrorMatrix(uniques, confusionMatrix);
            var error = Error(targets, predictions);

            return m_converter.Convert(uniques, confusionMatrix, errorMatrix, error);
        }
    }
}
