using System;
using System.Threading.Tasks;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;

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
        /// <param name="executor"></param>
        /// <param name="isTraining"></param>
        public static void Forward(Variable input,
            Variable Scale, Variable Bias,
            Variable BatchColumnMeans, Variable BatchcolumnVars,
            Variable MovingAverageMeans, Variable MovingAverageVariance,
            Executor executor, bool isTraining, Variable output)
        {
            var src = executor.GetTensor(input);
            var dst = executor.GetTensor(output);

            var scaleData = executor.GetTensor(Scale).Data;
            var biasData = executor.GetTensor(Bias).Data;

            var bColumnMeans = executor.GetTensor(BatchColumnMeans).Data;
            var bcolumnVars = executor.GetTensor(BatchcolumnVars).Data;
            var movingAverageMeans = executor.GetTensor(MovingAverageMeans).Data;
            var movingAverageVariance = executor.GetTensor(MovingAverageVariance).Data;

            if(input.DimensionCount == 4)
            {
                Forward4D(isTraining, src, dst, scaleData, biasData, 
                    bColumnMeans, bcolumnVars, movingAverageMeans, movingAverageVariance);
            }
            else if(input.DimensionCount == 2)
            {
                Forward2D(isTraining, src, dst, scaleData, biasData,
                    bColumnMeans, bcolumnVars, movingAverageMeans, movingAverageVariance);
            }
            else
            {
                throw new ArgumentException("Expected 2D or 4D input and output");
            }
        }

        static void Forward4D(bool isTraining, Tensor<float> src, Tensor<float> dst, 
            float[] scaleData, float[] biasData, 
            float[] batchColumnMeansData, float[] batcColumnVarsData, 
            float[] movingAverageMeansData, float[] movingAverageVarianceData)
        {
            int N = src.Dimensions[0]; // number of items in mini batch
            int C = src.Dimensions[1];
            int H = src.Dimensions[2];
            int W = src.Dimensions[3];

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
                    mean = movingAverageMeansData[c];
                    variance = movingAverageVarianceData[c];
                }

                var scale = scaleData[c];
                var bias = biasData[c];


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
                    movingAverageMeansData[c] = MovingAverage(movingAverageMeansData[c], mean);
                    movingAverageVarianceData[c] = MovingAverage(movingAverageVarianceData[c], variance);

                    batchColumnMeansData[c] = mean;
                    batcColumnVarsData[c] = variance;
                }
            });
        }

        static void Forward2D(bool isTraining, Tensor<float> src, Tensor<float> dst,
            float[] scaleData, float[] biasData,
            float[] batchColumnMeansData, float[] batcColumnVarsData,
            float[] movingAverageMeansData, float[] movingAverageVarianceData)
        {
            int N = src.Dimensions[0]; // number of items in mini batch
            int C = src.Dimensions[1];

            double eps = 1e-6;

            var srcData = src.Data;
            var dstData = dst.Data;

            Parallel.For(0, C, c =>
            {
                float mean = 0;
                float variance = 0;

                if (isTraining)
                {
                    for (int n = 0; n < N; ++n)
                    {
                        var srcBOffSet = src.DimensionOffSets[0] * n;
                        var srcIndex = srcBOffSet + c;
                        mean += srcData[srcIndex];
                    }
                    mean /= C;

                    for (int n = 0; n < N; ++n)
                    {
                        var srcBOffSet = src.DimensionOffSets[0] * n;
                        var srcIndex = srcBOffSet + c;
                        var m = srcData[srcIndex] - mean;
                        variance += m * m;
                    }
                    variance = 1f / (float)Math.Sqrt(variance / C + eps);
                }
                else
                {
                    mean = movingAverageMeansData[c];
                    variance = movingAverageVarianceData[c];
                }

                var scale = scaleData[c];
                var bias = biasData[c];


                for (int n = 0; n < N; ++n)
                {
                    var bOffSet = src.DimensionOffSets[0] * n;
                    var index = bOffSet + c;
                    dstData[index] = scale * (srcData[index] - mean) * variance + bias;
                }
                if (isTraining)
                {
                    movingAverageMeansData[c] = MovingAverage(movingAverageMeansData[c], mean);
                    movingAverageVarianceData[c] = MovingAverage(movingAverageVarianceData[c], variance);

                    batchColumnMeansData[c] = mean;
                    batcColumnVarsData[c] = variance;
                }
            });
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="Scale"></param>
        /// <param name="Bias"></param>
        /// <param name="BatchColumnMeans"></param>
        /// <param name="BatchcolumnVars"></param>
        /// <param name="executor"></param>
        /// <param name="output"></param>
        public static void Backward(Variable input,
            Variable Scale, Variable Bias,
            Variable BatchColumnMeans, Variable BatchcolumnVars,
            Executor executor, Variable output)
        {
            var src = executor.GetTensor(input);
            var diff_src = executor.GetGradient(input);

            var scaleData = executor.GetTensor(Scale).Data;

            var scaleGradientData = executor.GetGradient(Scale).Data;
            var biasGradientData = executor.GetGradient(Bias).Data;

            var meanData = executor.GetTensor(BatchColumnMeans).Data;
            var varianceData = executor.GetTensor(BatchcolumnVars).Data;

            var diff_dst = executor.GetTensor(output);

            if (input.DimensionCount == 4)
            {
                Backward4D(src, diff_src, scaleData, 
                    scaleGradientData, biasGradientData, meanData, varianceData, diff_dst);
            }
            else if (input.DimensionCount == 2)
            {
                Backward2D(src, diff_src, scaleData, 
                    scaleGradientData, biasGradientData, meanData, varianceData, diff_dst);
            }
            else
            {
                throw new ArgumentException("Expected 2D or 4D input and output");
            }
        }

        static void Backward4D(Tensor<float> src, Tensor<float> diff_src, 
            float[] scaleData, float[] scaleGradientData, 
            float[] biasGradientData, 
            float[] meanData, float[] varianceData, 
            Tensor<float> diff_dst)
        {
            int N = src.Dimensions[0];
            int C = src.Dimensions[1];
            int H = src.Dimensions[2];
            int W = src.Dimensions[3];

            //double eps = conf_.desc()->batch_norm_epsilon;
            //bool use_scaleshift = conf_.use_scaleshift();
            //bool calculate_diff_stats = !conf_.omit_stats();

            //const float eps = 1e-6f;

            var srcData = src.Data;
            var diffDstData = diff_dst.Data;
            var diffSrcData = diff_src.Data;

            //#pragma omp parallel for schedule(static)
            for (int c = 0; c < C; ++c)
            {
                var v_mean = meanData[c];
                var v_variance = varianceData[c];
                //var sqrt_variance = 1.0f / (float)Math.Sqrt(v_variance + eps);
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

                diff_gamma *= v_variance;

                scaleGradientData[c] = diff_gamma;
                biasGradientData[c] = diff_gamma;

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
                                diff_gamma * v_variance / (W * H * N);

                            v_diff_src *= gamma * v_variance;
                            diffSrcData[diffSrcIndex] = v_diff_src;
                        }
                    }
                }
            }
        }

        static void Backward2D(Tensor<float> src, Tensor<float> diff_src,
             float[] scaleData, float[] scaleGradientData,
             float[] biasGradientData,
             float[] meanData, float[] varianceData,
             Tensor<float> diff_dst)
        {
            int N = src.Dimensions[0];
            int C = src.Dimensions[1];

            //double eps = conf_.desc()->batch_norm_epsilon;
            //bool use_scaleshift = conf_.use_scaleshift();
            //bool calculate_diff_stats = !conf_.omit_stats();

            //const float eps = 1e-6f;

            var srcData = src.Data;
            var diffDstData = diff_dst.Data;
            var diffSrcData = diff_src.Data;

            for (int c = 0; c < C; ++c)
            {
                var v_mean = meanData[c];
                var v_variance = varianceData[c];
                //var sqrt_variance = 1.0f / (float)Math.Sqrt(v_variance + eps);
                var gamma = scaleData[c];
                var diff_gamma = 0.0f;
                var diff_beta = 0.0f;

                for (int n = 0; n < N; ++n)
                {
                    var srcBOffSet = src.DimensionOffSets[0] * n;
                    var diffDstBOffSet = diff_dst.DimensionOffSets[0] * n;

                    var srcIndex = srcBOffSet + c;
                    var diffDstIndex = diffDstBOffSet + c;

                    diff_gamma += (srcData[srcIndex] - v_mean)
                        * diffDstData[diffDstIndex];
                    diff_beta += diffDstData[diffDstIndex];
                }

                diff_gamma *= v_variance;

                scaleGradientData[c] = diff_gamma;
                biasGradientData[c] = diff_gamma;

                for (int n = 0; n < N; ++n)
                {
                    var srcBOffSet = src.DimensionOffSets[0] * n;
                    var diffDstBOffSet = diff_dst.DimensionOffSets[0] * n;
                    var diffSrcBOffSet = diff_src.DimensionOffSets[0] * n;

                    var srcIndex = srcBOffSet + c;
                    var diffDstIndex = diffDstBOffSet + c;
                    var diffSrcIndex = diffSrcBOffSet + c;

                    var v_diff_src = diffDstData[diffDstIndex];

                    v_diff_src -= diff_beta / (C) + (srcData[srcIndex] - v_mean) *
                        diff_gamma * v_variance / (C);

                    v_diff_src *= gamma * v_variance;
                    diffSrcData[diffSrcIndex] = v_diff_src;
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
