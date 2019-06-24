namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    /// <summary>
    /// Classification impurity calculator using the Gini impurity.
    /// </summary>
    public sealed class GiniClassificationImpurityCalculator : ClassificationImpurityCalculator, IImpurityCalculator
    {
        /// <summary>
        /// 
        /// </summary>
        public GiniClassificationImpurityCalculator()
        {
        }

        /// <summary>
        /// Gets the unique target names
        /// </summary>
        public double[] TargetNames
        {
            get { return m_targetNames; }
        }

        /// <summary>
        /// Calculates child impurities with current split index
        /// </summary>
        /// <returns></returns>
        public override ChildImpurities ChildImpurities()
        {
            var giniLeft = 0.0;
            var giniRight = 0.0;

            foreach (var targetValue in m_targetNames)
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

            foreach (var targetValue in m_targetNames)
            {
                var value = m_weightedTargetCount[(int)targetValue];
                gini += value * value;
            }

            gini = 1.0 - gini / (m_weightedTotal * m_weightedTotal);

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
            var leftImpurity = (m_weightedLeft / m_weightedTotal) * childImpurities.Left;
            var rightImpurity = (m_weightedRight / m_weightedTotal) * childImpurities.Right;

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

            foreach (var targetValue in m_targetNames)
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

        /// <summary>
        /// Laplace adjusted probabilities. Same order as m_uniqueTargetNames
        /// </summary>
        /// <returns></returns>
        public override double[] LeafProbabilities()
        {
            var probabilities = new double[m_targetNames.Length];
            var probabilityFactor = 1.0 / (m_weightedTotal + m_targetNames.Length);

            for (int i = 0; i < m_targetNames.Length; i++)
            {
                int targetValue = (int)m_targetNames[i];
                var targetProbability = (m_weightedTargetCount[targetValue] + 1) * probabilityFactor;
                probabilities[i] = targetProbability;
            }

            return probabilities;
        }
    }
}
