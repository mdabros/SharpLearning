using System;
using System.Threading.Tasks;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class BatchNormalization
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
        public static void Forward(Tensor<float> input, 
            Tensor<float> Scale, Tensor<float> Bias,
            float[] BatchColumnMeans, float[] BatchcolumnVars,
            float[] MovingAverageMeans, float[] MovingAverageVariance,
            Tensor <float> output, bool isTraining)
        {
            var src = input.AsTensor4D();
            var dst = output.AsTensor4D();
            
            int N = src.N; // number of items in mini batch
            int C = src.C;
            int H = src.H;
            int W = src.W;

            var sc = Scale.AsTensor1D();
            var bi = Scale.AsTensor1D();

            double eps = 1e-6;

            Parallel.For(0, C, c =>
            {
                float mean = 0;
                float variance = 0;

                if (isTraining)
                {
                    for (int n = 0; n < N; ++n)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w)                                
                                mean += src.At(n, c, h, w);

                    mean /= W * N * H;

                    for (int n = 0; n < N; ++n)
                        for (int h = 0; h < H; ++h)
                        {
                            for (int w = 0; w < W; ++w)
                            {
                                var m = src.At(n, c, h, w) - mean;
                                variance += m * m;
                            }
                        }
                    variance = 1f / (float)Math.Sqrt(variance / (W * H * N) + eps);
                }
                else
                {
                    mean = MovingAverageMeans[c];
                    variance = MovingAverageVariance[c];
                }

                var scale = sc.At(c);
                var bias = bi.At(c);


                for (int n = 0; n < N; ++n)
                    for (int h = 0; h < H; ++h)
                    {
                        for (int w = 0; w < W; ++w)
                        {
                            var value = scale * (src.At(n, c, h, w) - mean) * variance + bias;
                            dst.At(n, c, h, w, value);
                        }
                    }
                if (isTraining)
                {
                    MovingAverageMeans[c] = MovingAverage(MovingAverageMeans[c], mean);
                    MovingAverageVariance[c] = MovingAverage(MovingAverageVariance[c], variance);

                    BatchColumnMeans[c] = mean;
                    BatchcolumnVars[c] = variance;
                }
            });
        }

        static float MovingAverage(float currentValue, float value, float momentum = 0.99f)
        {
            var newValue = currentValue * momentum + value * (1.0f - momentum);
            return newValue;
        }
    }
}
