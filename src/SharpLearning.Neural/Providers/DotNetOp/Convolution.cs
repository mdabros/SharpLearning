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
            var N = input.Dimensions[0];
            var C = input.Dimensions[1];
            var H = input.Dimensions[2];
            var W = input.Dimensions[3];

            var src = storage.GetTensor(input);
            var dst = storage.GetTensor(output);
            var i2c = storage.GetTensor(im2Col);

            var w = storage.GetTensor(weights);
            var b = storage.GetTensor(bias);

            // Arrange input item for GEMM version of convolution.
            Im2Col(src, desc, borderMode, i2c);
            
            var filterGridWidth = ConvUtils.GetFilterGridLength(W, desc.FilterW, desc.StrideW, desc.PadW, borderMode);
            var filterGridHeight = ConvUtils.GetFilterGridLength(H, desc.FilterH, desc.StrideH, desc.PadH, borderMode);
            var filterCubeSize = C * desc.FilterW * desc.FilterH;
            var filterGridSize = filterGridWidth * filterGridHeight;

            var dstShape = dst.Shape;
            dst.Reshape(new TensorShape(desc.FilterCount, filterGridSize * N));

            // matrix multiplication for convolution
            w.Multiply(i2c, dst);
            dst.AddColumnWise(b.Data, dst);

            // reshape for output
            dst.Reshape(dstShape);
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
            var N = input.Dimensions[0];
            var C = input.Dimensions[1];
            var H = input.Dimensions[2];
            var W = input.Dimensions[3];

            var src = storage.GetTensor(input);
            var srcDiff = storage.GetGradient(input);
            var dst = storage.GetTensor(output);
            var dstDiff = storage.GetGradient(output);

            var i2c = storage.GetTensor(im2Col);

            var w = storage.GetTensor(weights);
            var wDiff = storage.GetGradient(weights);
            var b = storage.GetTensor(bias);
            var bDiff = storage.GetGradient(bias);

            // Arrange input item for GEMM version of convolution.
            var filterGridWidth = ConvUtils.GetFilterGridLength(W, desc.FilterW, desc.StrideW, desc.PadW, borderMode);
            var filterGridHeight = ConvUtils.GetFilterGridLength(H, desc.FilterH, desc.StrideH, desc.PadH, borderMode);
            var filterCubeSize = C * desc.FilterW * desc.FilterH;
            var filterGridSize = filterGridWidth * filterGridHeight;

            var dstDiffShape = dstDiff.Shape;
            dstDiff.Reshape(new TensorShape(desc.FilterCount, filterGridSize * N));
            
            // Calculate gradients for weights and biases
            dstDiff.TransposeAndMultiply(i2c, wDiff);
            dstDiff.SumRows(bDiff.Data);

            // calcualte delta for next layer.
            w.TransposeThisAndMultiply(dstDiff, i2c);

            // convert back to original layout
            Col2Im(i2c, desc, borderMode, srcDiff);

            // reshape dstDiff to original layout.
            dstDiff.Reshape(dstDiffShape);
        }

        /// <summary>
        /// transform from tensor: [filterCount, BatchSize, GridsizeH, GridSizeW)]
        /// to tensor:             [batchSize, filterCount, GridSizeH, GridSizeW)]
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dst"></param>
        public static void SwitchDimensionOneAndTwo(Tensor<double> src, Tensor<double> dst)
        {
            var N = dst.Dimensions[0]; // BatchSize
            var C = dst.Dimensions[1]; // filterCount
            var H = dst.Dimensions[2]; // GridsizeH
            var W = dst.Dimensions[3]; // GridsizeW

            var dstData = dst.Data;
            var srcData = src.Data;

            for (int n = 0; n < N; n++)
            {
                for (int c = 0; c < C; c++)
                {
                    for (int h = 0; h < H; h++)
                    {
                        for (int w = 0; w < W; w++)
                        {
                            var srcIndex = Index(c, n, h, w, src);
                            var dstIndex = Index(n, c, h, w, dst);

                            dstData[dstIndex] = srcData[srcIndex];
                        }
                    }
                }
            }
        }

        static int Index(int n, int c, int h, int w, Tensor<double> t)
        {
            var index = t.DimensionOffSets[0] * n + t.DimensionOffSets[1] * c + t.DimensionOffSets[2] * h + w;
            return index;
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

            var outputIndex = 0;
            //Parallel.For(0, N, n =>
            for (int n = 0; n  < N; n ++)
            {
                var imOffSetB = im.DimensionOffSets[0] * n;
                //var outputIndex = im2Col.DimensionOffSets[0] * n;

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
            }//);
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


            var outputIndex = 0;

            //Parallel.For(0, N, n =>
            for (int n = 0; n < N; n++)
            {
                var imOffSetB = im.DimensionOffSets[0] * n;
                //var outputIndex = im2Col.DimensionOffSets[0] * n;

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
            }//);
        }
    }
}
