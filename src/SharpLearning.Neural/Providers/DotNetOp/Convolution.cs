using System;
using System.Numerics;
using System.Threading.Tasks;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Providers.DotNetOp
{
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
        /// <param name="desc"></param>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        /// <param name="borderMode"></param>
        /// <param name="output"></param>
        /// <param name="storage"></param>
        public static void Forward(Variable input, 
            Variable im2Col, Conv2DDescriptor desc, 
            Variable weights, Variable bias, BorderMode borderMode,
            Variable output, NeuralNetStorage storage)
        {
            var src = storage.GetTensor(input);
            var dst = storage.GetTensor(output);
            var i2c = storage.GetTensor(im2Col);

            var w = storage.GetTensor(weights);
            var b = storage.GetTensor(bias);

            // Arrange input item for GEMM version of convolution.
            Im2Col(src, desc, borderMode, i2c);

            // matrix multiplication for convolution
            w.Multiply(i2c, dst);
            dst.AddColumnWise(b.Data, dst);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="im2Col"></param>
        /// <param name="desc"></param>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        /// <param name="borderMode"></param>
        /// <param name="output"></param>
        /// <param name="storage"></param>
        public static void Backward(Variable input,
            Variable im2Col, Conv2DDescriptor desc,
            Variable weights, Variable bias, BorderMode borderMode,
            Variable output, NeuralNetStorage storage)
        {
            var src = storage.GetTensor(input);
            var srcDiff = storage.GetGradient(input);
            var dst = storage.GetTensor(output);
            var dstDiff = storage.GetGradient(output);

            var i2c = storage.GetTensor(im2Col);

            var w = storage.GetTensor(weights);
            var wDiff = storage.GetGradient(weights);
            var b = storage.GetTensor(bias);
            var bDiff = storage.GetGradient(bias);

            // Calculate gradients for weights and biases
            dstDiff.TransposeAndMultiply(i2c, wDiff);
            dstDiff.SumRows(bDiff.Data);

            // calcualte delta for next layer.
            w.TransposeThisAndMultiply(dstDiff, i2c);
            // convert back to original layout
            Col2Im(i2c, desc, borderMode, srcDiff);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="im"></param>
        /// <param name="desc"></param>
        /// <param name="borderMode"></param>
        /// <param name="im2Col"></param>
        public static void Im2Col(Tensor<double> im, Conv2DDescriptor desc, BorderMode borderMode, Tensor<double> im2Col)
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

            Parallel.For(0, N, n =>
            //for (int n = 0; n  < N; n ++)
            {
                var imOffSetB = im.DimensionOffSets[0] * n;
                var outputIndex = im2Col.DimensionOffSets[0] * n;

                for (int c = 0; c < channels_col; ++c)
                {
                    int offsetW = c % filterW;
                    int offsetH = (c / filterW) % filterH;
                    int imC = c / filterH / filterW;

                    var imOffSetC = imOffSetB + im.DimensionOffSets[1] * imC;
                    for (int h = 0; h < filterGridHeight; h++)
                    {
                        for (int w = 0; w < filterGridWidth; w++)
                        {
                            int h_pad = h * strideH - padH + offsetH;
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
        /// 
        /// </summary>
        /// <param name="im2Col"></param>
        /// <param name="desc"></param>
        /// <param name="borderMode"></param>
        /// <param name="im"></param>
        public static void Col2Im(Tensor<double> im2Col, Conv2DDescriptor desc, BorderMode borderMode, Tensor<double> im)
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

            Parallel.For(0, N, n =>
            //for (int n = 0; n < N; n++)
            {
                var imOffSetB = im.DimensionOffSets[0] * n;
                var outputIndex = im2Col.DimensionOffSets[0] * n;

                for (int c = 0; c < channels_col; ++c)
                {
                    int offsetW = c % filterW;
                    int offsetH = (c / filterW) % filterH;
                    int imC = c / filterH / filterW;

                    var imOffSetC = imOffSetB + im.DimensionOffSets[1] * imC;

                    for (int h = 0; h < filterGridHeight; ++h)
                    {
                        for (int w = 0; w < filterGridWidth; ++w)
                        {
                            int h_pad = h * strideH - padH + offsetH;
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
