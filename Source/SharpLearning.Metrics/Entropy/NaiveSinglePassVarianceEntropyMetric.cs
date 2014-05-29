
namespace SharpLearning.Metrics.Entropy
{
    public sealed class NaiveSinglePassVarianceEntropyMetric : IEntropyMetric
    {
        /// <summary>
        /// Calculates the variance of a sample. Main use is for decision tree regression
        /// http://en.wikipedia.org/wiki/Decision_tree_learning
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public double Entropy(double[] values)
        {
            var n = 0;
            var sum = 0.0;
            var sumSqr = 0.0;

            for (int i = 0; i < values.Length; i++)
            {
                var x = values[i];
                ++n;
                sum += x;
                sumSqr += x * x;
            }
            int divisor = n - 1;
            if (divisor == 0) { return 0.0; }

            var variance = (sumSqr - ((sum * sum) / n)) / divisor;
            return variance;
        }
    }
}
