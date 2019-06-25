using System;
using System.Linq;
using System.Text;

namespace SharpLearning.Metrics
{
    /// <summary>
    /// McNemar test for comparing two models. 
    /// The important part of the comparison is the number of times model1 is right where model2 is wrong and vice-versa.
    /// A clear improvement between two models would be if this number is, say 1 to 10.
    /// https://en.wikipedia.org/wiki/McNemar%27s_test
    /// </summary>
    public sealed class McNemarModelComparison
    {
        /// <summary>
        /// Compares two model using the McNemar test.
        /// The important part of the comparison is the number of times model1 is right where model2 is wrong and vice-versa.
        /// A clear improvement between two models would be if this number is, say 1 to 10. The resulting matrix is of format
        ///             Model1Wrong Model1Right
        /// Model2Wrong     x            y
        /// Model2Right     z            j
        /// </summary>
        /// This is also visible if outputting the CompareString.
        /// <param name="model1Predictions"></param>
        /// <param name="model2Predictions"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public int[][] Compare(double[] model1Predictions, double[] model2Predictions, double[] targets)
        {
            if (model1Predictions.Length != model2Predictions.Length || 
                model1Predictions.Length != targets.Length)
            {
                throw new ArgumentException("Model prediction lengths differ from target length. " +
                    $"Model1: {model1Predictions.Length}, " + 
                    $"Model2: {model2Predictions.Length}, " + 
                    $"Targets: {targets.Length}");
            }

            var modelCount = 2;
            var mcNemarmatrix = new int[modelCount][].Select(s => new int[modelCount]).ToArray();

            for (int i = 0; i < targets.Length; i++)
            {
                var model1Prediction = model1Predictions[i];
                var model2Prediction = model2Predictions[i];
                var target = targets[i];

                if (model1Prediction == target && model2Prediction == target)
                {
                    mcNemarmatrix[1][1]++;
                }
                else if (model1Prediction != target && model2Prediction != target)
                {
                    mcNemarmatrix[0][0]++;
                }
                else if (model1Prediction != target && model2Prediction == target)
                {
                    mcNemarmatrix[1][0]++;
                }
                else if (model1Prediction == target && model2Prediction != target)
                {
                    mcNemarmatrix[0][1]++;
                }
            }

            return mcNemarmatrix;
        }

        /// <summary>
        /// Outputs a string representation of a McNemar test.
        /// The important part of the comparison is the number of times model1 is right where model2 is wrong and vice-versa.
        /// A clear improvement between two models would be if this number is, say 1 to 10. The resulting matrix is of format
        ///             Model1Wrong Model1Right
        /// Model2Wrong     x            y
        /// Model2Right     z            j
        /// </summary>
        /// <param name="model1Predictions"></param>
        /// <param name="model2Predictions"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public string CompareString(double[] model1Predictions, double[] model2Predictions, double[] targets)
        {
            var mcNemarMatrix = Compare(model1Predictions, model2Predictions, targets);
            
            var builder = new StringBuilder();
            builder.AppendLine(";Model1Wrong;Model1Right");
            builder.AppendLine($"Model2Wrong;{mcNemarMatrix[0][0]};{mcNemarMatrix[0][1]}");
            builder.Append($"Model2Right;{mcNemarMatrix[1][0]};{mcNemarMatrix[1][1]}");
            
            return builder.ToString();
        }
    }
}
