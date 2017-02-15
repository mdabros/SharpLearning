using System;
using System.Collections.Generic;

namespace SharpLearning.Metrics.Ranking
{
    /// <summary>
    /// Calculates the average precision ranking metric
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class AveragePrecisionRankingMetric<T> : IRankingMetric<T>
    {
        readonly int m_k;
        readonly HashSet<T> m_workTargets = new HashSet<T>();

        /// <summary>
        /// Takes the top k predictions to consider
        /// </summary>
        /// <param name="k"></param>
        public AveragePrecisionRankingMetric(int k)
        {
            if (k < 1) { throw new ArgumentException("k must be at least 1"); }
            m_k = k;
        }

        /// <summary>
        /// Calculates the average precision error 1 - average precision
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Error(T[] targets, T[] predictions)
        {
            var length = m_k;
            if (predictions.Length < length)
                length = predictions.Length;

            m_workTargets.Clear();
            foreach (var target in targets)
            {
                m_workTargets.Add(target);
            }

            var score = 0.0;
            var hits = 0.0;

            for (int i = 0; i < length; i++)
            {
                var prediction = predictions[i];
                if(m_workTargets.Contains(prediction) &&
                    !Contains(predictions, i, prediction))
                {
                    hits += 1.0;
                    score += hits / (i + 1.0);
                }
            }

            double minLength = Math.Min(targets.Length, m_k);

            return 1.0 - score / minLength;
        }

        bool Contains(T[] predictions, int i, T prediction)
        {
            var predictionFound = false;
            for (int j = 0; j < i; j++)
            {
                if (predictions[j].Equals(prediction))
                {
                    predictionFound = true;
                }
            }
            return predictionFound;
        }
    }
}
