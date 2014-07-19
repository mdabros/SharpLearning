using SharpLearning.Containers.Views;
using System;
using System.Linq;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    /// <summary>
    /// Base class for classifiction impurity calculators
    /// </summary>
    public abstract class ClasificationImpurityCalculator
    {
        Interval1D m_interval;
        int m_currentPosition;

        protected double m_weightedInterval = 0.0;
        protected double m_weightedLeft = 0.0;
        protected double m_weightedRight = 0.0;

        protected double[] m_weightedTargetCount;
        protected double[] m_weightedTargetCountLeft;
        protected double[] m_weightedTargetCountRight;

        protected double[] m_targets;
        protected double[] m_weights;

        protected double[] m_uniqueTargets;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="uniqueTargets"></param>
        /// <param name="targets"></param>
        /// <param name="weights"></param>
        /// <param name="interval"></param>
        public ClasificationImpurityCalculator(double[] uniqueTargets, double[] targets, double[] weights, Interval1D interval)
        {
            if (targets == null) { throw new ArgumentException("targets"); }
            if (weights == null) { throw new ArgumentException("weights"); }
            if (uniqueTargets == null) { throw new ArgumentException("uniqueTargets"); }
            m_targets = targets;
            m_weights = weights;
            m_uniqueTargets = uniqueTargets;
            m_interval = interval;

            var maxIndex = (int)m_uniqueTargets.Max() + 1;
            m_weightedTargetCount = new double[maxIndex];
            m_weightedTargetCountLeft = new double[maxIndex];
            m_weightedTargetCountRight = new double[maxIndex];

            var w = 1.0;
            var weightsPresent = m_weights.Length != 0;

            for (int i = m_interval.FromInclusive; i < m_interval.ToExclusive; i++)
            {
                if (weightsPresent)
                    w = weights[i];

                var targetIndex = (int)targets[i];
                m_weightedTargetCount[targetIndex] += w;

                m_weightedInterval += w;
            }

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
            m_weightedRight = m_weightedInterval;

            Array.Clear(m_weightedTargetCountLeft, 0, m_weightedTargetCountLeft.Length);
            Array.Copy(m_weightedTargetCount, m_weightedTargetCountRight, m_weightedTargetCount.Length);
        }

        /// <summary>
        /// Updates impurity calculator with new split index
        /// </summary>
        /// <param name="newPosition"></param>
        public void Update(int newPosition)
        {
            if(m_currentPosition > newPosition)
            {
                throw new ArgumentException("New position: " + newPosition +
                    " must be larget than current: " + m_currentPosition);
            }

            var weightsPresent = m_weights.Length != 0;
            var w = 1.0;
            var w_diff = 0.0;

            for (int i = m_currentPosition; i < newPosition; i++)
            {
                if (weightsPresent)
                    w = m_weights[i];

                var targetIndex = (int)m_targets[i];
                m_weightedTargetCountLeft[targetIndex] += w;
                m_weightedTargetCountRight[targetIndex] -= w;

                w_diff += w;
            }

            m_weightedLeft += w_diff;
            m_weightedRight -= w_diff;

            m_currentPosition = newPosition;
        }

        /// <summary>
        /// Calculates child impurities with current split index
        /// </summary>
        /// <returns></returns>
        public abstract ChildImpurities ChildImpurities();

        /// <summary>
        /// Calculate the node impurity
        /// </summary>
        /// <returns></returns>
        public abstract double NodeImpurity();

        /// <summary>
        /// Calculates the impurity improvement at the current split index
        /// </summary>
        /// <param name="impurity"></param>
        /// <returns></returns>
        public abstract double ImpurityImprovement(double impurity);

        /// <summary>
        /// Calculates the weighted leaf value
        /// </summary>
        /// <returns></returns>
        public abstract double LeafValue();
    }
}
