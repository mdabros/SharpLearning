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
        public static string Convert<T>(
            List<T> uniqueTargets, 
            Dictionary<T, string> targetStringMapping, 
            int[,] confusionMatrix, 
            double[,] errorMatrix, 
            double error)
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
        public static string Convert<T>(
            List<T> uniqueTargets, 
            int[,] confusionMatrix, 
            double[,] errorMatrix, 
            double error)
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
            var rows = combinedMatrix.GetLength(0);
            var cols = combinedMatrix.GetLength(1);

            for (int r = 0; r < rows; r++)
            {
                var row = string.Format("{0}", uniqueTargets[r]);
                for (int c = 0; c < cols; c++)
                {
                    row += string.Format(";{0:0.000}", combinedMatrix[r, c]);
                }
                builder.AppendLine(row);
            }

            builder.AppendLine(string.Format("Error: {0:0.000}", 100.0 * error));

            return builder.ToString();
        }

        static double[,] CombineMatrices(int[,] classCountMatrix, double[,] classErrorMatrix)
        {
            var rows = classErrorMatrix.GetLength(0);
            var cols = classErrorMatrix.GetLength(1);
            var combinedMatrix = new double[rows, 2 * cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    combinedMatrix[r, c] = classCountMatrix[r, c];
                }

                for (int c = 0; c < cols; c++)
                {
                    combinedMatrix[r, c + cols] = classErrorMatrix[r, c] * 100.0; // convert to percentage
                }
            }
            return combinedMatrix;
        }
    }
}
