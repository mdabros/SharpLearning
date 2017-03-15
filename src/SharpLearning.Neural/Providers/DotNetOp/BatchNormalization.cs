using System;
using System.Threading.Tasks;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public class BatchNormalization
    {
        /// <summary>
        /// 
        /// </summary>
        public BatchNormalization()
        {
        }

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
        public void Forward(Tensor<float> input, 
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
        public void Backward(Tensor<float> input,
            Tensor<float> Scale, Tensor<float> Bias,
            Tensor<float> ScaleGradients, Tensor<float> BiasGradients,
            float[] BatchColumnMeans, float[] BatchcolumnVars,
            Tensor<float> dstDiff, Tensor<float> srcDiff)
        {
            var src = input;
            var mean = BatchColumnMeans;
            var variance = BatchcolumnVars;
            var diff_dst = dstDiff;
            var diff_src = srcDiff;

            int N = input.Dimensions[0];
            int C = input.Dimensions[1];
            int H = input.Dimensions[2];
            int W = input.Dimensions[3];

            var scaleData = Scale.Data;

            var scaleGradientsData = ScaleGradients.Data;
            var biasGradientsData = BiasGradients.Data;

            //double eps = conf_.desc()->batch_norm_epsilon;
            //bool use_scaleshift = conf_.use_scaleshift();
            //bool calculate_diff_stats = !conf_.omit_stats();

            const float eps = 1e-6f;

            var srcData = src.Data;
            var diffDstData = diff_dst.Data;
            var diffSrcData = diff_src.Data;

            //#pragma omp parallel for schedule(static)
            for (int c = 0; c < C; ++c)
            {
                var v_mean = mean[c];
                var v_variance = variance[c];
                var sqrt_variance = 1.0f / (float)Math.Sqrt(v_variance + eps);
                var gamma = scaleData[c];
                var diff_gamma = 0.0f;
                var diff_beta = 0.0f;

                var srcCoffSet = src.DimensionOffSets[1] * c;
                var diffDstCOffSet = diff_dst.DimensionOffSets[1] * c;
                var diffSrcCOffSet = diff_src.DimensionOffSets[1] * c;

                for (int n = 0; n < N; ++n)
                {
                    var srcBOffSet = src.DimensionOffSets[0] * n;
                    var diffDstBOffSet = diff_dst.DimensionOffSets[0] * n;

                    for (int h = 0; h < H; ++h)
                    {
                        var srcHOffSet = src.DimensionOffSets[2] * h;
                        var diffDstHOffSet = diff_dst.DimensionOffSets[2] * h;

                        for (int w = 0; w < W; ++w)
                        {
                            var srcIndex = srcBOffSet + srcCoffSet + srcHOffSet + w;
                            var diffDstIndex = diffDstBOffSet + diffDstCOffSet + diffDstHOffSet + w;

                            diff_gamma += (srcData[srcIndex] - v_mean)
                                * diffDstData[diffDstIndex];
                            diff_beta += diffDstData[diffDstIndex];
                        }
                    }
                }

                diff_gamma *= sqrt_variance;

                scaleGradientsData[c] = diff_gamma;
                biasGradientsData[c] = diff_gamma;

                for (int n = 0; n < N; ++n)
                {
                    var srcBOffSet = src.DimensionOffSets[0] * n;
                    var diffDstBOffSet = diff_dst.DimensionOffSets[0] * n;
                    var diffSrcBOffSet = diff_src.DimensionOffSets[0] * n;

                    for (int h = 0; h < H; ++h)
                    {
                        var srcHOffSet = src.DimensionOffSets[2] * h;
                        var diffDstHOffSet = diff_dst.DimensionOffSets[2] * h;
                        var diffSrcHOffSet = diff_src.DimensionOffSets[2] * h;

                        for (int w = 0; w < W; ++w)
                        {
                            var srcIndex = srcBOffSet + srcCoffSet + srcHOffSet + w;
                            var diffDstIndex = diffDstBOffSet + diffDstCOffSet + diffDstHOffSet + w;
                            var diffSrcIndex = diffSrcBOffSet + diffSrcCOffSet + diffSrcHOffSet + w;

                            var v_diff_src = diffDstData[diffDstIndex];
                            
                            v_diff_src -= diff_beta / (W * H * N) +
                                (srcData[srcIndex] - v_mean) *
                                diff_gamma * sqrt_variance / (W * H * N);

                            v_diff_src *= gamma * sqrt_variance;
                            diffSrcData[diffSrcIndex] = v_diff_src;
                        }
                    }
                }
            }
        }

        static float MovingAverage(float currentValue, float value, float momentum = 0.99f)
        {
            var newValue = currentValue * momentum + value * (1.0f - momentum);
            return newValue;
        }
    }
}
