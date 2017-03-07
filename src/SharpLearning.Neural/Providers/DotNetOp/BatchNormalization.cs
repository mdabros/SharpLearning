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
            if(input.DimensionCount != 4 || output.DimensionCount != 4)
            {
                throw new ArgumentException("Expected 4-dimensional input and output");
            }

            var src = input;
            var dst = output;
            
            int N = src.Dimensions[0]; // number of items in mini batch
            int C = src.Dimensions[1];
            int H = src.Dimensions[2];
            int W = src.Dimensions[3];

            var sc = Scale.Data;
            var bi = Bias.Data;

            double eps = 1e-6;

            var srcData = src.Data;
            var dstData = dst.Data;

            Parallel.For(0, C, c =>
            {
                float mean = 0;
                float variance = 0;

                var srcCoffSet = src.DimensionOffSets[1] * c;

                if (isTraining)
                {
                    for (int n = 0; n < N; ++n)
                    {
                        var srcBOffSet = src.DimensionOffSets[0] * n;
                        for (int h = 0; h < H; ++h)
                        {
                            var srcHOffSet = src.DimensionOffSets[2] * h;
                            for (int w = 0; w < W; ++w)
                            {
                                var srcIndex = srcBOffSet + srcCoffSet + srcHOffSet + w;
                                mean += srcData[srcIndex];
                            }
                        }
                    }
                    mean /= W * N * H;

                    for (int n = 0; n < N; ++n)
                    {
                        var srcBOffSet = src.DimensionOffSets[0] * n;
                        for (int h = 0; h < H; ++h)
                        {
                            var srcHOffSet = src.DimensionOffSets[2] * h;
                            for (int w = 0; w < W; ++w)
                            {
                                var srcIndex = srcBOffSet + srcCoffSet + srcHOffSet + w;
                                var m = srcData[srcIndex] - mean;
                                variance += m * m;
                            }
                        }
                    }
                    variance = 1f / (float)Math.Sqrt(variance / (W * H * N) + eps);
                }
                else
                {
                    mean = MovingAverageMeans[c];
                    variance = MovingAverageVariance[c];
                }

                var scale = sc[c];
                var bias = bi[c];


                for (int n = 0; n < N; ++n)
                {
                    var bOffSet = src.DimensionOffSets[0] * n;
                    for (int h = 0; h < H; ++h)
                    {
                        var hOffSet = src.DimensionOffSets[2] * h;
                        for (int w = 0; w < W; ++w)
                        {
                            var index = bOffSet + srcCoffSet + hOffSet + w;
                            dstData[index] = scale * (srcData[index] - mean) * variance + bias;
                        }
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
