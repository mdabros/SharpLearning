using SharpLearning.Containers.Views;
using SharpLearning.Containers;

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

        /// <summary>
        /// Calculates the variance of a sample over the provided interval. Main use is for decision tree regression
        /// http://en.wikipedia.org/wiki/Decision_tree_learning
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public double Entropy(double[] values, Interval1D interval)
        {
            var n = 0;
            var sum = 0.0;
            var sumSqr = 0.0;

            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
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


        public double Entropy(double[] values, double[] weights, Interval1D interval)
        {
            var weightSum = 0.0;
            var weightSumSqr = 0.0;
            var mean = 0.0;

            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                var w = weights[i];
                var x = values[i];
                weightSum += w;
                weightSumSqr += w * w;
                mean += x * w; 
            }

            if (weightSum == 0.0)
                return 0.0;

            mean = mean / weightSum;
            var meanSum = 0.0;

            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
			{
                var w = weights[i];
                var x = values[i];
                meanSum += w * (x - mean) * (x - mean);     			    
			}

            var divisor = (weightSum * weightSum - weightSumSqr);
            
            if (divisor == 0.0)
                return 0.0;

            var variance = (weightSum / divisor) * meanSum;

            return variance;
        }
    }
}
