using System;
using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    /// <summary>
    /// Base class for classification impurity calculators
    /// </summary>
    public abstract class ClassificationImpurityCalculator
    {
        /// <summary>
        /// 
        /// </summary>
        protected Interval1D m_interval;

        /// <summary>
        /// 
        /// </summary>
        protected int m_currentPosition;

        /// <summary>
        /// 
        /// </summary>
        protected double m_weightedTotal = 0.0;

        /// <summary>
        /// 
        /// </summary>
        protected double m_weightedLeft = 0.0;

        /// <summary>
        /// 
        /// </summary>
        protected double m_weightedRight = 0.0;

        internal TargetCounts m_weightedTargetCount = new TargetCounts();
        internal TargetCounts m_weightedTargetCountLeft = new TargetCounts();
        internal TargetCounts m_weightedTargetCountRight = new TargetCounts();

        /// <summary>
        /// 
        /// </summary>
        protected double[] m_targets;

        /// <summary>
        /// 
        /// </summary>
        protected double[] m_weights;

        /// <summary>
        /// 
        /// </summary>
        protected double[] m_targetNames;

        /// <summary>
        /// 
        /// </summary>
        protected int m_maxTargetNameIndex;

        /// <summary>
        /// 
        /// </summary>
        protected int m_targetIndexOffSet;

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
        public ClassificationImpurityCalculator()
        {
        }

        /// <summary>
        /// Initialize the calculator with targets, weights and work interval
        /// </summary>
        /// <param name="targetNames"></param>
        /// <param name="targets"></param>
        /// <param name="weights"></param>
        /// <param name="interval"></param>
        public void Init(double[] targetNames, double[] targets, double[] weights, Interval1D interval)
        {
            m_targets = targets ?? throw new ArgumentException(nameof(targets));
            m_weights = weights ?? throw new ArgumentException(nameof(weights));
            m_targetNames = targetNames ?? throw new ArgumentException(nameof(targetNames));
            m_interval = interval;

            SetMinMaxTargetNames();
            if(m_targetIndexOffSet > 0)
            {
                m_targetIndexOffSet = 0;
            }
            else
            {
                m_targetIndexOffSet = m_targetIndexOffSet * -1; 
            }

            m_weightedTargetCount.Reset(m_maxTargetNameIndex, m_targetIndexOffSet);
            m_weightedTargetCountLeft.Reset(m_maxTargetNameIndex, m_targetIndexOffSet);
            m_weightedTargetCountRight.Reset(m_maxTargetNameIndex, m_targetIndexOffSet);

            var w = 1.0;
            var weightsPresent = m_weights.Length != 0;

            m_weightedTotal = 0.0;
            m_weightedLeft = 0.0;
            m_weightedRight = 0.0;

            for (int i = m_interval.FromInclusive; i < m_interval.ToExclusive; i++)
            {
                if (weightsPresent)
                    w = weights[i];

                var targetIndex = (int)targets[i];
                m_weightedTargetCount[targetIndex] += w;

                m_weightedTotal += w;
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
            m_weightedRight = m_weightedTotal;

            m_weightedTargetCountLeft.Clear();
            m_weightedTargetCountRight.SetCounts(m_weightedTargetCount);
        }

        /// <summary>
        /// Updates impurities according to the new interval
        /// </summary>
        /// <param name="newInterval"></param>
        public void UpdateInterval(Interval1D newInterval)
        {
            Init(m_targetNames, m_targets, m_weights, newInterval);
        }


        /// <summary>
        /// Updates impurity calculator with new split index
        /// </summary>
        /// <param name="newPosition"></param>
        public void UpdateIndex(int newPosition)
        {
            if(m_currentPosition > newPosition)
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

        /// <summary>
        /// Calculates the weighted leaf value
        /// </summary>
        /// <returns></returns>
        public abstract double[] LeafProbabilities();

        void SetMinMaxTargetNames()
        {
            m_maxTargetNameIndex = int.MinValue;
            m_targetIndexOffSet = int.MaxValue;

            foreach (int value in m_targetNames)
            {
                if (value > m_maxTargetNameIndex)
                {
                    m_maxTargetNameIndex = value;
                }
                else if (value < m_targetIndexOffSet)
                {
                    m_targetIndexOffSet = value;
                }
            }

            m_maxTargetNameIndex = m_maxTargetNameIndex + 1;
        }
    }
}
