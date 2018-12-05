using System;

namespace SharpLearning.Neural.Test.RefactorBranStorm
{
    interface ILoss
    {
        Tensor<float> SampleLosses(Tensor<float> targets, Tensor<float> predictions);
        float AccumulateSampleLoss(Tensor<float> sampleLosses);
    }

    class MeanSquareLoss : ILoss
    {
        public Tensor<float> SampleLosses(Tensor<float> targets, Tensor<float> predictions)
        {
            CheckDimensions(targets, predictions);

            var losses = new Tensor<float>(targets.Dimensions, DataLayout.RowMajor);

            for (int i = 0; i < targets.Data.Length; i++)
            {
                losses.Data[i] = predictions.Data[i] - targets.Data[i];
            }

            return losses;
        }

        public float AccumulateSampleLoss(Tensor<float> sampleLosses)
        {
            var accumulatedLoss = 0.0f;

            for (int i = 0; i < sampleLosses.Data.Length; i++)
            {
                var sampleLoss = sampleLosses.Data[i];
                accumulatedLoss += sampleLoss * sampleLoss;
            }

            accumulatedLoss = accumulatedLoss / sampleLosses.Data.Length;

            return accumulatedLoss;
        }

        static void CheckDimensions(Tensor<float> targets, Tensor<float> predictions)
        {
            if (targets.Shape != predictions.Shape)
            {
                throw new ArgumentException($"Target shape: {targets.Shape} differs from "
                    + $" Prediction shape: {predictions.Shape}");
            }
        }
    }
}
