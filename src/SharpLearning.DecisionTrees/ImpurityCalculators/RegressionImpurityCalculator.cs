using System;
using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    /// <summary>
    /// Regression impurity calculator using variance and friedmans
    /// calculation for impurity improvement.
    /// </summary>
    public sealed class RegressionImpurityCalculator : IImpurityCalculator
    {
        Interval1D m_interval;
        int m_currentPosition;

        double m_weightedTotal = 0.0;
        double m_weightedLeft = 0.0;
        double m_weightedRight = 0.0;

        double m_meanLeft = 0.0;
        double m_meanRight = 0.0;
        double m_meanTotal = 0.0;
               
        double m_sqSumLeft = 0.0;
        double m_sqSumRight = 0.0;
        double m_sqSumTotal = 0.0;
               
        double m_varLeft = 0.0;
        double m_varRight = 0.0;
       
        double m_sumLeft = 0.0;
        double m_sumRight = 0.0;
        double m_sumTotal = 0.0;
        
        double[] m_targets;
        double[] m_weights;

        /// <summary>
        /// 
        /// </summary>
        public double WeightedLeft { get { return m_weightedLeft; } }

        /// <summary>
        /// 
        /// </summary>
        public double WeightedRight { get { return m_weightedRight; } }

        /// <summary>
        /// 
        /// </summary>
        public RegressionImpurityCalculator()
        {
        }

        /// <summary>
        /// Initialize the calculator with targets, weights and work interval 
        /// </summary>
        /// <param name="uniqueTargets"></param>
        /// <param name="targets"></param>
        /// <param name="weights"></param>
        /// <param name="interval"></param>
        public void Init(double[] uniqueTargets, double[] targets, double[] weights, Interval1D interval)
        {
            m_targets = targets ?? throw new ArgumentException(nameof(targets));
            m_weights = weights ?? throw new ArgumentException(nameof(weights));
            m_interval = interval;

            m_weightedTotal = 0.0;
            m_weightedLeft = 0.0;
            m_weightedRight = 0.0;

            m_meanLeft = 0.0;
            m_meanRight = 0.0;
            m_meanTotal = 0.0;

            m_sqSumLeft = 0.0;
            m_sqSumRight = 0.0;
            m_sqSumTotal = 0.0;

            m_varRight = 0.0;
            m_varLeft = 0.0;

            m_sumLeft = 0.0;
            m_sumRight = 0.0;
            m_sumTotal = 0.0;

            var w = 1.0;
            var weightsPresent = m_weights.Length != 0;

            for (int i = m_interval.FromInclusive; i < m_interval.ToExclusive; i++)
            {
                if (weightsPresent)
                    w = weights[i];

                var targetValue = targets[i];
                var wTarget = w * targetValue;
                m_sumTotal += wTarget;
                m_sqSumTotal += wTarget * targetValue;

                m_weightedTotal += w;
            }

            m_meanTotal = m_sumTotal / m_weightedTotal;

            m_currentPosition = m_interval.FromInclusive;
            this.Reset();
        }

        /// <summary>
        /// Resets impurity calculator
        /// </summary>
        public void Reset()
        {
            m_currentPosition = m_interval.FromInclusive;

            m_weightedLeft = 0.0;
            m_weightedRight = m_weightedTotal;

            m_meanRight = m_meanTotal;
            m_meanLeft = 0.0;
            m_sumRight = m_sqSumTotal;
            m_sqSumLeft = 0.0;

            m_varRight = (m_sqSumRight / m_weightedTotal -
                m_meanRight * m_meanRight);
            m_varLeft = 0.0;

            m_sumRight = m_sumTotal;
            m_sumLeft = 0.0;
        }

        /// <summary>
        /// Updates impurities according to the new interval
        /// </summary>
        /// <param name="newInterval"></param>
        public void UpdateInterval(Interval1D newInterval)
        {
            Init(new double[0], m_targets, m_weights, newInterval);
        }

        /// <summary>
        /// Updates impurity calculator with new split index
        /// </summary>
        /// <param name="newPosition"></param>
        public void UpdateIndex(int newPosition)
        {
            if (m_currentPosition > newPosition)
            {
                throw new ArgumentException("New position: " + newPosition +
                    " must be larger than current: " + m_currentPosition);
            }

            var weightsPresent = m_weights.Length != 0;
            var w = 1.0;
            var w_diff = 0.0;
            
            for (int i = m_currentPosition; i < newPosition; i++)
            {
                if (weightsPresent)
                    w = m_weights[i];

                var targetValue = m_targets[i];
                var wTarget = w * targetValue;

                m_sumLeft += wTarget;
                m_sumRight -= wTarget;

                var wTargetSq = wTarget * targetValue;

                m_sqSumLeft += wTargetSq;
                m_sqSumRight -= wTargetSq;

                w_diff += w;
            }

            m_weightedLeft += w_diff;
            m_weightedRight -= w_diff;

            m_meanLeft = m_sumLeft / m_weightedLeft;
            m_meanRight = m_sumRight / m_weightedRight;

            m_varLeft = (m_sqSumLeft / m_weightedLeft -
                m_meanLeft * m_meanLeft);

            m_varRight = (m_sqSumRight / m_weightedRight -
                m_meanRight * m_meanRight);

            m_currentPosition = newPosition;
        }

        /// <summary>
        /// Calculate the node impurity
        /// </summary>
        /// <returns></returns>
        public double NodeImpurity()
        {
            var impurity = (m_sqSumTotal / m_weightedTotal -
                m_meanTotal * m_meanTotal);

            return impurity;
        }

        /// <summary>
        /// Calculates child impurities with current split index
        /// </summary>
        /// <returns></returns>
        public ChildImpurities ChildImpurities()
        {
            return new ChildImpurities(m_varLeft, m_varRight);
        }

        /// <summary>
        /// Calculates the impurity improvement at the current split index
        /// </summary>
        /// <param name="impurity"></param>
        /// <returns></returns>
        public double ImpurityImprovement(double impurity)
        {
            var diff = ((m_sumLeft / m_weightedLeft) -
                (m_sumRight / m_weightedRight));

            var improvement = (m_weightedLeft * m_weightedRight * diff * diff /
                (m_weightedLeft + m_weightedRight));

            return improvement;
        }

        /// <summary>
        /// Calculates the weighted leaf value
        /// </summary>
        /// <returns></returns>
        public double LeafValue()
        {
            return m_meanTotal;
        }

        /// <summary>
        /// Unique target names are not available for regression
        /// </summary>
        public double[] TargetNames
        {
            get { return new double[0]; }
        }

        /// <summary>
        /// Probabilities are not available for regression
        /// </summary>
        /// <returns></returns>
        public double[] LeafProbabilities()
        {
            return new double[0];
        }
    }
}
