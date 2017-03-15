using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Providers
{
    /// <summary>
    /// 
    /// </summary>
    public interface IBatchNormalization
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="Scale"></param>
        /// <param name="Bias"></param>
        /// <param name="BatchColumnMeans"></param>
        /// <param name="BatchcolumnVars"></param>
        /// <param name="MovingAverageMeans"></param>
        /// <param name="MovingAverageVariance"></param>
        /// <param name="output"></param>
        /// <param name="isTraining"></param>
        void Forward(Tensor<float> input,
            Tensor<float> Scale, Tensor<float> Bias,
            float[] BatchColumnMeans, float[] BatchcolumnVars,
            float[] MovingAverageMeans, float[] MovingAverageVariance,
            Tensor<float> output, bool isTraining);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="Scale"></param>
        /// <param name="Bias"></param>
        /// <param name="ScaleGradients"></param>
        /// <param name="BiasGradients"></param>
        /// <param name="BatchColumnMeans"></param>
        /// <param name="BatchcolumnVars"></param>
        /// <param name="dstDiff"></param>
        /// <param name="srcDiff"></param>
        void Backward(Tensor<float> input,
            Tensor<float> Scale, Tensor<float> Bias,
            Tensor<float> ScaleGradients, Tensor<float> BiasGradients,
            float[] BatchColumnMeans, float[] BatchcolumnVars,
            Tensor<float> dstDiff, Tensor<float> srcDiff);
    }
}