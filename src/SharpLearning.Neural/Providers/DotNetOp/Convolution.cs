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
        /// <param name="conv"></param>
        /// <param name="desc"></param>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        /// <param name="borderMode"></param>
        /// <param name="output"></param>
        /// <param name="storage"></param>
        public static void Forward(Variable input, 
            Variable im2Col, Variable conv,
            Conv2DDescriptor desc, 
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
            w.Multiply(i2c, co);
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
            Conv2DDescriptor desc,
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
            co.TransposeAndMultiply(i2c, wDiff);
            co.SumRows(bDiff.Data);

            // calcualte delta for next layer.
            w.TransposeThisAndMultiply(co, i2c);

            // convert back to original layout
            Col2Im(i2c, desc, borderMode, srcDiff);
        }

        /// <summary>
        /// transform from tensor: [C, N, H, W)]
        /// to tensor:             [N, C, H, W)]
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dst"></param>
        public static void SwitchDimensionOneAndTwo(Tensor<double> src, Tensor<double> dst)
        {
            var N = dst.Dimensions[0]; 
            var C = dst.Dimensions[1];
            var H = dst.Dimensions[2]; 
            var W = dst.Dimensions[3]; 

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

        static bool is_a_ge_zero_and_a_lt_b(int a, int b)
        {
            return (uint)(a) < (uint)(b);
            //return static_cast<unsigned>(a) < static_cast<unsigned>(b);
        }

        /// <summary>
        /// based on https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
        /// </summary>
        /// <param name="im"></param>
        /// <param name="desc"></param>
        /// <param name="borderMode"></param>
        /// <param name="im2Col"></param>
        public static void im2col_cpu(Tensor<double> im, Conv2DDescriptor desc, BorderMode borderMode, Tensor<double> im2Col)
        {
            int N = im.Dimensions[0];
            int channels = im.Dimensions[1];
            int height = im.Dimensions[2];
            int width = im.Dimensions[3];
            int kernel_h = desc.FilterH;
            int kernel_w = desc.FilterW;
            int pad_h = desc.PadH;
            int pad_w = desc.PadW;
            int stride_h = desc.StrideH;
            int stride_w = desc.StrideW;
            int dilation_h = 1;
            int dilation_w = 1;

            var data_im = im.Data;
            var data_col = im2Col.Data;

            int output_h = ConvUtils.GetFilterGridLength(height, kernel_h, stride_h, pad_h, borderMode);
            int output_w = ConvUtils.GetFilterGridLength(width, kernel_w, stride_w, pad_w, borderMode);

            //int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
            //int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
            int channel_size = height * width;

            var outputIndex = 0;

            for (int n = 0; n < N; n++)
            {
                var nOffSet = im.DimensionOffSets[0] * n;
                for (int channel = 0; channel < channels; channel++)//data_im += channel_size)
                {
                    var cOffSet = nOffSet + im.DimensionOffSets[1] * channel;
                    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
                    {
                        //var hOffSet = cOffSet + im.DimensionOffSets[2] * kernel_row;
                        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
                        {
                            //var wOffSet = hOffSet + kernel_row;
                            var input_row = -pad_h + kernel_row * dilation_h;
                            for (int output_rows = output_h - 1; output_rows >= 0; output_rows--)
                            {
                                if (!is_a_ge_zero_and_a_lt_b(input_row, height))
                                {
                                    for (int output_cols = output_w - 1; output_cols >= 0; output_cols--)
                                    {
                                        //*(data_col++) = 0;
                                        data_col[outputIndex++] = 0;
                                    }
                                }
                                else
                                {
                                    int input_col = -pad_w + kernel_col * dilation_w;
                                    for (int output_col = output_w - 1; output_col >= 0; output_col--)
                                    {
                                        if (is_a_ge_zero_and_a_lt_b(input_col, width))
                                        {
                                            //*(data_col++) = data_im[input_row * width + input_col];
                                            data_col[outputIndex++] = data_im[cOffSet + input_row * width + input_col];
                                        }
                                        else
                                        {
                                            //*(data_col++) = 0;
                                            data_col[outputIndex++] = 0;
                                        }
                                        input_col += stride_w;
                                    }
                                }
                                input_row += stride_h;
                            }
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
        /// based on https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
        /// </summary>
        /// <param name="im2Col"></param>
        /// <param name="desc"></param>
        /// <param name="borderMode"></param>
        /// <param name="im"></param>
        public static void col2im_cpu(Tensor<double> im2Col, Conv2DDescriptor desc, BorderMode borderMode, Tensor<double> im)
        {

            int N = im.Dimensions[0];
            int channels = im.Dimensions[1];
            int height = im.Dimensions[2];
            int width = im.Dimensions[3];
            int kernel_h = desc.FilterH;
            int kernel_w = desc.FilterW;
            int pad_h = desc.PadH;
            int pad_w = desc.PadW;
            int stride_h = desc.StrideH;
            int stride_w = desc.StrideW;
            int dilation_h = 1;
            int dilation_w = 1;

            var data_im = im.Data;
            var data_col = im2Col.Data;

            int output_h = ConvUtils.GetFilterGridLength(height, kernel_h, stride_h, pad_h, borderMode);
            int output_w = ConvUtils.GetFilterGridLength(width, kernel_w, stride_w, pad_w, borderMode);
            //caffe_set(height * width * channels, Dtype(0), data_im);

            //int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
            //int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
            int channel_size = height * width;

            var im2ColIndex = 0;
            for (int n = 0; n < N; n++)
            {
                var nOffSet = im.DimensionOffSets[0] * n;
                for (int channel = 0; channel < channels; channel++)//data_im += channel_size)
                //for (int channel = channels; channel--; data_im += channel_size)
                {
                    var cOffSet = nOffSet + im.DimensionOffSets[1] * channel;
                    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
                    {
                        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
                        {
                            int input_row = -pad_h + kernel_row * dilation_h;
                            for (int output_rows = output_h - 1; output_rows >= 0; output_rows--)
                            //for (int output_rows = output_h; output_rows; output_rows--)
                            {
                                if (!is_a_ge_zero_and_a_lt_b(input_row, height))
                                {
                                    im2ColIndex += output_w;
                                }
                                else
                                {
                                    int input_col = -pad_w + kernel_col * dilation_w;
                                    for (int output_col = output_w - 1; output_col >= 0; output_col--)
                                    //for (int output_col = output_w; output_col; output_col--)
                                    {
                                        if (is_a_ge_zero_and_a_lt_b(input_col, width))
                                        {
                                            data_im[cOffSet + input_row * width + input_col] += data_col[im2ColIndex];
                                        }
                                        im2ColIndex++;
                                        input_col += stride_w;
                                    }
                                }
                                input_row += stride_h;
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Based on https://github.com/NVIDIA/torch-cunn/blob/master/lib/THCUNN/im2col.h
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
