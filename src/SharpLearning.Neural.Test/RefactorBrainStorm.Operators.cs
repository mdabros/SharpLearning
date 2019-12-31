using System;
using System.Threading.Tasks;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.Neural.Test.RefactorBranStorm
{
    public static class Operators
    {
        public static class ReLU
        {
            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="output"></param>
            /// <param name="storage"></param>
            public static void Forward(Variable input, Variable output, NeuralNetStorage storage)
            {
                var inputTensor = storage.GetTensor(input);
                var outputTensor = storage.GetTensor(output);

                for (int j = 0; j < inputTensor.Data.Length; j++)
                {
                    outputTensor.Data[j] = ReluMax(inputTensor.Data[j]);
                }
            }


            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="output"></param>
            /// <param name="strorage"></param>
            public static void Backward(Variable input, Variable output, NeuralNetStorage strorage)
            {
                var outputTensor = strorage.GetTensor(output);
                var outputGradient = strorage.GetGradient(output);
                var inputGradient = strorage.GetGradient(input);

                for (int j = 0; j < outputTensor.Data.Length; j++)
                {
                    inputGradient.Data[j] = Derivative(outputTensor.Data[j], outputGradient.Data[j]);
                }
            }

            static float ReluMax(float input)
            {
                return Math.Max(0, input);
            }

            static float Derivative(float output, float outputGradient)
            {
                if (output > 0.0)
                    return outputGradient;
                else
                    return 0.0f;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public static class Sigmoid
        {
            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="output"></param>
            /// <param name="storage"></param>
            public static void Forward(Variable input, Variable output, NeuralNetStorage storage)
            {
                var src = storage.GetTensor(input).Data;
                var dst = storage.GetTensor(output).Data;

                for (int j = 0; j < src.Length; j++)
                {
                    dst[j] = DoSigmoid(src[j]);
                }
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="output"></param>
            /// <param name="strorage"></param>
            public static void Backward(Variable input, Variable output, NeuralNetStorage strorage)
            {
                var dst = strorage.GetTensor(output).Data;
                var dstDiff = strorage.GetGradient(output).Data;
                var srcDiff = strorage.GetGradient(input).Data;

                for (int j = 0; j < dst.Length; j++)
                {
                    srcDiff[j] = Derivative(dst[j], dstDiff[j]);
                }
            }

            static float DoSigmoid(float input)
            {
                return 1.0f / (1.0f + (float)Math.Exp(-input));
            }

            static float Derivative(float dst, float dstGradient)
            {
                return dst * (1.0f - dst) * dstGradient;
            }
        }

        public static class Dense
        {
            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="weights"></param>
            /// <param name="bias"></param>
            /// <param name="output"></param>
            /// <param name="storage"></param>
            public static void Forward(Variable input,
                Variable weights, Variable bias,
                Variable output, NeuralNetStorage storage)
            {
                var src = storage.GetTensor(input);

                var w = storage.GetTensor(weights);
                var b = storage.GetTensor(bias).Data;

                var dst = storage.GetTensor(output);

                src.Multiply(w, dst);
                dst.AddRowWise(b, dst);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="weights"></param>
            /// <param name="bias"></param>
            /// <param name="output"></param>
            /// <param name="storage"></param>
            public static void Backward(Variable input,
                Variable weights, Variable bias,
                Variable output, NeuralNetStorage storage)
            {
                var src = storage.GetTensor(input);
                var srcDiff = storage.GetGradient(input);

                var w = storage.GetTensor(weights);
                var wDiff = storage.GetGradient(weights);

                var bDiff = storage.GetGradient(bias).Data;
                var dstDiff = storage.GetGradient(output);

                // calculate gradients
                src.TransposeThisAndMultiply(dstDiff, wDiff);
                dstDiff.SumColumns(bDiff);

                // calculate delta for next layer
                dstDiff.TransposeAndMultiply(w, srcDiff);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public struct ConvolutionDescriptor
        {
            /// <summary>
            /// 
            /// </summary>
            public readonly int FilterChannels;
            /// <summary>
            /// 
            /// </summary>
            public readonly int FilterH;
            /// <summary>
            /// 
            /// </summary>
            public readonly int FilterW;
            /// <summary>
            /// 
            /// </summary>
            public readonly int StrideH;
            /// <summary>
            /// 
            /// </summary>
            public readonly int StrideW;
            /// <summary>
            /// 
            /// </summary>
            public readonly int PadH;
            /// <summary>
            /// 
            /// </summary>
            public readonly int PadW;

            /// <summary>
            /// 
            /// </summary>
            /// <param name="filterChannels">Number of filters</param>
            /// <param name="filterH">The height of each filter</param>
            /// <param name="filterW">The width of each filter</param>
            /// <param name="strideH">The vertical stride of the filter</param>
            /// <param name="strideW">The horizontal stride of the filter</param>
            /// <param name="padH">Zero padding at the top and bottom</param>
            /// <param name="padW">Zero padding to the left and right</param>
            public ConvolutionDescriptor(int filterChannels, int filterH, int filterW,
                int strideH, int strideW,
                int padH, int padW)
            {
                if (filterChannels < 1)
                { throw new ArgumentException($"filterChannels must be at least 1, was {filterChannels}"); }
                if (filterH < 1)
                { throw new ArgumentException($"filterH must be at least 1, was {filterH}"); }
                if (filterW < 1)
                { throw new ArgumentException($"filterW must be at least 1, was {filterW}"); }
                if (strideH < 1)
                { throw new ArgumentException($"strideH must be at least 1, was {strideH}"); }
                if (strideW < 1)
                { throw new ArgumentException($"strideW must be at least 1, was {strideW}"); }
                if (padH < 0)
                { throw new ArgumentException($"padH must be at least 0, was {padH}"); }
                if (padW < 0)
                { throw new ArgumentException($"padW must be at least 0, was {padW}"); }

                FilterChannels = filterChannels;
                FilterH = filterH;
                FilterW = filterW;
                StrideH = strideH;
                StrideW = strideW;
                PadH = padH;
                PadW = padW;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public static class Convolution
        {
            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="im2Col"></param>
            /// <param name="conv"></param>
            /// <param name="desc"></param>
            /// <param name="weights"></param>
            /// <param name="bias"></param>
            /// <param name="borderMode"></param>
            /// <param name="output"></param>
            /// <param name="storage"></param>
            public static void Forward(Variable input,
                Variable im2Col, Variable conv,
                ConvolutionDescriptor desc,
                Variable weights, Variable bias, BorderMode borderMode,
                Variable output, NeuralNetStorage storage)
            {
                var N = input.Dimensions[0];
                var C = input.Dimensions[1];
                var H = input.Dimensions[2];
                var W = input.Dimensions[3];

                var src = storage.GetTensor(input);
                var dst = storage.GetTensor(output);
                var i2c = storage.GetTensor(im2Col);
                var co = storage.GetTensor(conv);

                var w = storage.GetTensor(weights);
                var b = storage.GetTensor(bias);

                // Arrange input item for GEMM version of convolution.
                Im2Col(src, desc, borderMode, i2c);

                // matrix multiplication for convolution
                w.TransposeAndMultiply(i2c, co);
                co.AddColumnWise(b.Data, co);

                // switch dimension one and two to get correct layout for next layer.
                SwitchDimensionOneAndTwo(co, dst);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="im2Col"></param>
            /// <param name="conv"></param>
            /// <param name="desc"></param>
            /// <param name="weights"></param>
            /// <param name="bias"></param>
            /// <param name="borderMode"></param>
            /// <param name="output"></param>
            /// <param name="storage"></param>
            public static void Backward(Variable input,
                Variable im2Col, Variable conv,
                ConvolutionDescriptor desc,
                Variable weights, Variable bias, BorderMode borderMode,
                Variable output, NeuralNetStorage storage)
            {
                var N = input.Dimensions[0];
                var C = input.Dimensions[1];
                var H = input.Dimensions[2];
                var W = input.Dimensions[3];

                var src = storage.GetTensor(input);
                var srcDiff = storage.GetGradient(input);
                var dst = storage.GetTensor(output);
                var dstDiff = storage.GetGradient(output);

                var i2c = storage.GetTensor(im2Col);
                var co = storage.GetTensor(conv);

                var w = storage.GetTensor(weights);
                var wDiff = storage.GetGradient(weights);
                var b = storage.GetTensor(bias);
                var bDiff = storage.GetGradient(bias);

                // Switch dimension one and two to have correct layout for GEMM version of convolution
                SwitchDimensionOneAndTwo(dstDiff, co);

                // Calculate gradients for weights and biases
                co.Multiply(i2c, wDiff);
                co.SumRows(bDiff.Data);

                // calcualte delta for next layer.
                co.TransposeThisAndMultiply(w, i2c);

                // convert back to original layout
                Col2Im(i2c, desc, borderMode, srcDiff);
            }

            /// <summary>
            /// transform from tensor: [C, N, H, W)]
            /// to tensor:             [N, C, H, W)]
            /// </summary>
            /// <param name="src"></param>
            /// <param name="dst"></param>
            public static void SwitchDimensionOneAndTwo(Tensor<float> src, Tensor<float> dst)
            {
                var N = dst.Dimensions[0];
                var C = dst.Dimensions[1];
                var H = dst.Dimensions[2];
                var W = dst.Dimensions[3];

                var dstData = dst.Data;
                var srcData = src.Data;

                for (int n = 0; n < N; n++)
                {
                    var dstOffSetB = dst.DimensionOffSets[0] * n;
                    for (int c = 0; c < C; c++)
                    {
                        var srcOffSetC = src.DimensionOffSets[0] * c + src.DimensionOffSets[1] * n;
                        var dstOffSetC = dstOffSetB + dst.DimensionOffSets[1] * c;

                        for (int h = 0; h < H; h++)
                        {
                            var srcOffSetH = srcOffSetC + src.DimensionOffSets[2] * h;
                            var dstOffSetH = dstOffSetC + dst.DimensionOffSets[2] * h;

                            for (int w = 0; w < W; w++)
                            {
                                var srcIndex = srcOffSetH + w;//Index(c, n, h, w, src);
                                var dstIndex = dstOffSetH + w;//Index(n, c, h, w, dst);

                                dstData[dstIndex] = srcData[srcIndex];
                            }
                        }
                    }
                }
            }

            /// <summary>
            /// Based on https://github.com/NVIDIA/torch-cunn/blob/master/lib/THCUNN/im2col.h
            /// </summary>
            /// <param name="im"></param>
            /// <param name="desc"></param>
            /// <param name="borderMode"></param>
            /// <param name="im2Col"></param>
            public static void Im2Col(Tensor<float> im, ConvolutionDescriptor desc, BorderMode borderMode, Tensor<float> im2Col)
            {
                var N = im.Dimensions[0];
                var C = im.Dimensions[1];
                var H = im.Dimensions[2];
                var W = im.Dimensions[3];

                int filterW = desc.FilterW;
                int filterH = desc.FilterH;
                int strideH = desc.StrideH;
                int strideW = desc.StrideW;
                int padH = desc.PadH;
                int padW = desc.PadW;

                var filterGridWidth = ConvUtils.GetFilterGridLength(W, filterW, strideW, padW, borderMode);
                var filterGridHeight = ConvUtils.GetFilterGridLength(H, filterH, strideH, padH, borderMode);
                int channels_col = C * filterH * filterW;

                var imData = im.Data;
                var im2ColData = im2Col.Data;

                var filterCubeSize = C * filterH * filterW;
                var filterGridSize = filterGridWidth * filterGridHeight;

                var im2ColBatchSize = filterGridSize * filterCubeSize;

                Parallel.For(0, N, n =>
                {
                    var imOffSetB = im.DimensionOffSets[0] * n;
                    var outputIndex = im2ColBatchSize * n;

                    for (int c = 0; c < channels_col; ++c)
                    {
                        int offsetW = c % filterW;
                        int offsetH = (c / filterW) % filterH;
                        int imC = c / filterH / filterW;

                        var imOffSetC = imOffSetB + im.DimensionOffSets[1] * imC;
                        for (int h = 0; h < filterGridHeight; h++)
                        {
                            int h_pad = h * strideH - padH + offsetH;

                            for (int w = 0; w < filterGridWidth; w++)
                            {
                                int w_pad = w * strideW - padW + offsetW;

                                var inputIndex = imOffSetC + h_pad * W + w_pad;

                                if (h_pad >= 0 && h_pad < H && w_pad >= 0 && w_pad < W)
                                {
                                    im2ColData[outputIndex++] = imData[inputIndex];
                                }
                                else
                                {
                                    im2ColData[outputIndex++] = 0;
                                }
                            }
                        }
                    }
                });
            }

            /// <summary>
            /// Based on https://github.com/NVIDIA/torch-cunn/blob/master/lib/THCUNN/im2col.h
            /// </summary>
            /// <param name="im2Col"></param>
            /// <param name="desc"></param>
            /// <param name="borderMode"></param>
            /// <param name="im"></param>
            public static void Col2Im(Tensor<float> im2Col, ConvolutionDescriptor desc, BorderMode borderMode, Tensor<float> im)
            {
                var N = im.Dimensions[0];
                var C = im.Dimensions[1];
                var H = im.Dimensions[2];
                var W = im.Dimensions[3];

                int filterW = desc.FilterW;
                int filterH = desc.FilterH;
                int strideH = desc.StrideH;
                int strideW = desc.StrideW;
                int padH = desc.PadH;
                int padW = desc.PadW;

                var filterGridWidth = ConvUtils.GetFilterGridLength(W, filterW, strideW, padW, borderMode);
                var filterGridHeight = ConvUtils.GetFilterGridLength(H, filterH, strideH, padH, borderMode);
                int channels_col = C * filterH * filterW;

                var imData = im.Data;
                var im2ColData = im2Col.Data;

                imData.Clear();

                var filterCubeSize = C * filterH * filterW;
                var filterGridSize = filterGridWidth * filterGridHeight;
                var im2ColBatchSize = filterGridSize * filterCubeSize;

                Parallel.For(0, N, n =>
                {
                    var imOffSetB = im.DimensionOffSets[0] * n;
                    var outputIndex = im2ColBatchSize * n;

                    for (int c = 0; c < channels_col; ++c)
                    {
                        int offsetW = c % filterW;
                        int offsetH = (c / filterW) % filterH;
                        int imC = c / filterH / filterW;

                        var imOffSetC = imOffSetB + im.DimensionOffSets[1] * imC;

                        for (int h = 0; h < filterGridHeight; ++h)
                        {
                            int h_pad = h * strideH - padH + offsetH;

                            for (int w = 0; w < filterGridWidth; ++w)
                            {
                                int w_pad = w * strideW - padW + offsetW;

                                if (h_pad >= 0 && h_pad < H && w_pad >= 0 && w_pad < W)
                                {
                                    var inputIndex = imOffSetC + h_pad * W + w_pad;
                                    imData[inputIndex] += im2ColData[outputIndex];
                                }
                                outputIndex++;
                            }
                        }
                    }
                });
            }
        }
    }
}
