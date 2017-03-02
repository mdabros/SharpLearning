using SharpLearning.Containers.Tensors;
using SharpLearning.Containers.Views;
using System;
using System.Threading.Tasks;

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
            var src = input.Indexer4D;
            var dst = output.Indexer4D;
            
            int N = src.DimNCount; // number of items in mini batch
            int C = src.DimZCount;
            int H = src.DimYCount;
            int W = src.DimXCount;

            double eps = 1e-6;

            Parallel.For(0, C, c =>
            {
                float mean = 0;
                float variance = 0;

                var interval = Interval1D.Create(0, W);
                var intervalValues = new float[interval.Length];

                if (isTraining)
                {
                    for (int n = 0; n < N; ++n)
                        for (int h = 0; h < H; ++h)
                        {
                            src.RangeX(h, c, n, interval, intervalValues);
                            for (int w = 0; w < W; ++w)
                                mean += intervalValues[w];//mean += src.At(w, h, c, n);
                        }

                    mean /= W * N * H;

                    for (int n = 0; n < N; ++n)
                        for (int h = 0; h < H; ++h)
                        {
                            src.RangeX(h, c, n, interval, intervalValues);
                            for (int w = 0; w < W; ++w)
                            {
                                var m = intervalValues[w] - mean;
                                //var m = src.At(w, h, c, n) - mean;
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

                var scale = Scale.Indexer1D.At(c);
                var bias = Bias.Indexer1D.At(c);


                for (int n = 0; n < N; ++n)
                    for (int h = 0; h < H; ++h)
                    {
                        src.RangeX(h, c, n, interval, intervalValues);
                        for (int w = 0; w < W; ++w)
                        {
                            var value = scale * (intervalValues[w] - mean) * variance + bias;
                            //var value = scale * (src.At(w, h, c, n) - mean) * variance + bias;
                            dst.At(w, h, c, n, value);
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
