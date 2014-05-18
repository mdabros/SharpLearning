using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Metrics.Classification
{
    public sealed class TotalErrorClassificationMetric : IClassificationMetric
    {
        readonly ClassificationMatrixStringConverter m_converter = new ClassificationMatrixStringConverter();
        readonly ClassificationMatrix m_classificationMatrix = new ClassificationMatrix();

        public double Error(double[] targets, double[] predictions)
        {
            var uniques = UniqueTargets(targets, predictions);

            var confusionMatrix = m_classificationMatrix.ConfusionMatrix(uniques, targets, predictions);
            var errorMatrix = m_classificationMatrix.ErrorMatrix(uniques, confusionMatrix);

            return TotalError(uniques, confusionMatrix);
        }

        double TotalError(List<double> uniques, int[][] confusionMatrix)
        {
            var totalSum = confusionMatrix.SelectMany(v => v).Sum();
            var errorSum = totalSum;

            for (int row = 0; row < uniques.Count; ++row)
            {
                errorSum -= confusionMatrix[row][row];
            }

            return (double)errorSum / (double)totalSum;
        }

        List<double> UniqueTargets(double[] targets, double[] predictions)
        {
            var uniquePredictions = predictions.Distinct();
            var uniqueTargets = targets.Distinct();
            var uniques = uniqueTargets.Union(uniquePredictions).ToList();

            uniques.Sort();
            return uniques;
        }

        public string ErrorString(double[] targets, double[] predictions)
        {
            var uniques = UniqueTargets(targets, predictions);

            var confusionMatrix = m_classificationMatrix.ConfusionMatrix(uniques, targets, predictions);
            var errorMatrix = m_classificationMatrix.ErrorMatrix(uniques, confusionMatrix);
            var error = TotalError(uniques, confusionMatrix);

            return m_converter.Convert(uniques, confusionMatrix, errorMatrix, error);
        }
    }
}
