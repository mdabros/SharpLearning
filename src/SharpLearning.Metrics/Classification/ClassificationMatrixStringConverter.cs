using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpLearning.Metrics.Classification
{
    /// <summary>
    /// 
    /// </summary>
    public static class ClassificationMatrixStringConverter
    {

        /// <summary>
        /// Creates a string representation of the classification matrix consisting of the provided confusion matrix and error matrix.
        /// Using the target naming provided in targetStringMapping.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="uniqueTargets"></param>
        /// <param name="targetStringMapping"></param>
        /// <param name="confusionMatrix"></param>
        /// <param name="errorMatrix"></param>
        /// <param name="error"></param>
        /// <returns></returns>
        public static string Convert<T>(List<T> uniqueTargets, Dictionary<T, string> targetStringMapping, 
            int[][] confusionMatrix, double[][] errorMatrix, double error)
        {
            var uniqueStringTargets = uniqueTargets.Select(t => targetStringMapping[t]).ToList();
            return Convert(uniqueStringTargets, confusionMatrix, errorMatrix, error);
        }
        
        /// <summary>
        /// Creates a string representation of the classification matrix consisting of the provided confusion matrix and error matrix
        /// </summary>
        /// <param name="uniqueTargets"></param>
        /// <param name="confusionMatrix"></param>
        /// <param name="errorMatrix"></param>
        /// <param name="error"></param>
        /// <returns></returns>
        public static string Convert<T>(List<T> uniqueTargets, int[][] confusionMatrix, double[][] errorMatrix, double error)
        {
            var combinedMatrix = CombineMatrices(confusionMatrix, errorMatrix);

            var builder = new StringBuilder();

            string horizontalClassNames = string.Empty;

            foreach (var className in uniqueTargets)
            {
                horizontalClassNames += string.Format(";{0}", className);
            }

            horizontalClassNames += horizontalClassNames;
            builder.AppendLine(horizontalClassNames);
            var numberofCols = combinedMatrix.First().Length;
            var numberOfRows = combinedMatrix.Length;

            for (int x = 0; x < numberOfRows; x++)
            {
                var row = string.Format("{0}", uniqueTargets[x]);
                for (int y = 0; y < numberofCols; y++)
                {
                    row += string.Format(";{0:0.000}", combinedMatrix[x][y]);
                }
                builder.AppendLine(row);
            }

            builder.AppendLine(string.Format("Error: {0:0.000}", 100.0 * error));

            return builder.ToString();
        }

        static double[][] CombineMatrices(int[][] classCountMatrix, double[][] classErrorMatrix)
        {
            var numberOfRows = classErrorMatrix.Length;
            var numberOfCols = classErrorMatrix.First().Length;
            var combinedMatrix = new double[classErrorMatrix.Length].Select(s => new double[2 * numberOfCols]).ToArray();
            for (int i = 0; i < numberOfRows; i++)
            {
                for (int j = 0; j < numberOfCols; j++)
                {
                    combinedMatrix[i][j] = classCountMatrix[i][j];
                }

                for (int j = 0; j < numberOfCols; j++)
                {
                    combinedMatrix[i][j + numberOfCols] = classErrorMatrix[i][j] * 100.0; // convert to percentage
                }
            }
            return combinedMatrix;
        }
    }
}
