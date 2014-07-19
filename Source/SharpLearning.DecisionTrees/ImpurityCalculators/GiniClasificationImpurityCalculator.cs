using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    /// <summary>
    /// Classifiction impurity calculator using the gini impurity.
    /// </summary>
    public sealed class GiniClasificationImpurityCalculator : ClasificationImpurityCalculator
    {
        public GiniClasificationImpurityCalculator(double[] uniqueTargets, double[] targets, double[] weights, Interval1D interval)
            : base(uniqueTargets, targets, weights, interval)
        {
        }

        /// <summary>
        /// Calculates child impurities with current split index
        /// </summary>
        /// <returns></returns>
        public override ChildImpurities ChildImpurities()
        {
            var giniLeft = 0.0;
            var giniRight = 0.0;

            foreach (var targetValue in m_uniqueTargets)
            {
                var targetIndex = (int)targetValue;
                var leftCount = m_weightedTargetCountLeft[targetIndex];
                var rightCount = m_weightedTargetCountRight[targetIndex];

                giniLeft += leftCount * leftCount;
                giniRight += rightCount * rightCount;
            }

            giniLeft = 1.0 - giniLeft / (m_weightedLeft * m_weightedLeft);
            giniRight = 1.0 - giniRight / (m_weightedRight * m_weightedRight);

            return new ChildImpurities(giniLeft, giniRight);
        }

        /// <summary>
        /// Calculate the node impurity
        /// </summary>
        /// <returns></returns>
        public override double NodeImpurity()
        {
            var gini = 0.0;

            foreach (var targetValue in m_uniqueTargets)
            {
                var value = m_weightedTargetCount[(int)targetValue];
                gini += value * value;
            }

            gini = 1.0 - gini / (m_weightedInterval * m_weightedInterval);

            return gini;
        }

        /// <summary>
        /// Calculates the impurity improvement at the current split index
        /// </summary>
        /// <param name="impurity"></param>
        /// <returns></returns>
        public override double ImpurityImprovement(double impurity)
        {
            var childImpurities = ChildImpurities();
            var leftImpurity = (m_weightedLeft / m_weightedInterval) * childImpurities.Left;
            var rightImpurity = (m_weightedRight / m_weightedInterval) * childImpurities.Right;

            return impurity - leftImpurity - rightImpurity;
        }

        /// <summary>
        /// Calculates the weighted leaf value
        /// </summary>
        /// <returns></returns>
        public override double LeafValue()
        {
            var bestTarget = 0.0;
            var maxWeight = 0.0;

            foreach (var targetValue in m_uniqueTargets)
            {
                var value = m_weightedTargetCount[(int)targetValue];
                if (value > maxWeight)
                {
                    maxWeight = value;
                    bestTarget = targetValue;
                }

            }

            return bestTarget;
        }
    }
}
