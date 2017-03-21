using MathNet.Numerics.LinearAlgebra;
using System;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Loss
{
    /// <summary>
    /// Log loss for neuralnet learner.
    /// This error metric is used when one needs to predict that something is true or false with a probability (likelihood) 
    /// ranging from definitely true (1) to equally true (0.5) to definitely false(0).
    /// The use of log on the error provides extreme punishments for being both confident and wrong.
    /// https://www.kaggle.com/wiki/LogarithmicLoss
    /// </summary>
    public sealed class LogLoss : ILoss
    {
        /// <summary>
        /// returns log los
        /// This error metric is used when one needs to predict that something is true or false with a probability (likelihood) 
        /// ranging from definitely true (1) to equally true (0.5) to definitely false(0).
        /// The use of log on the error provides extreme punishments for being both confident and wrong.
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public float Loss(Matrix<float> targets, Matrix<float> predictions)
        {
            return Loss(targets.Data(), predictions.Data(), targets.RowCount);
        }

        /// <summary>
        /// returns log los
        /// This error metric is used when one needs to predict that something is true or false with a probability (likelihood) 
        /// ranging from definitely true (1) to equally true (0.5) to definitely false(0).
        /// The use of log on the error provides extreme punishments for being both confident and wrong.
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public float Loss(Tensor<float> targets, Tensor<float> predictions)
        {
            return Loss(targets.Data, predictions.Data, targets.Dimensions[0]);
        }

        float Loss(float[] targetsData, float[] predictionsData, int batchSize)
        {
            var min = 1e-10f;
            var max = 1.0f - min;
            var sum = 0.0f;

            for (int i = 0; i < targetsData.Length; i++)
            {
                var clip = (float)Math.Log(Math.Max(min, Math.Min(predictionsData[i], max)));
                sum += targetsData[i] * clip;
            }

            return -sum / (float)batchSize;
        }
    }
}
