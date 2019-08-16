using System;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural
{
    /// <summary>
    /// Utiliy methods for convolutional operations in neural net layers.
    /// </summary>
    public static class ConvUtils
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="filterSize"></param>
        /// <param name="borderMode"></param>
        /// <returns></returns>
        public static int PaddingFromBorderMode(int filterSize, BorderMode borderMode)
        {
            switch (borderMode)
            {
                case BorderMode.Same:
                    return filterSize / 2;
                case BorderMode.Valid:
                    return 0;
                case BorderMode.Full:
                    return filterSize - 1;
                case BorderMode.Undefined:
                    return 0;
                default:
                    throw new ArgumentException("Unsupported border mode: " + borderMode);
            }
        }

        /// <summary>
        /// Gets the filter grid width based on input length, filter size, stride and padding.
        /// </summary>
        /// <param name="inputLength"></param>
        /// <param name="filterSize"></param>
        /// <param name="stride"></param>
        /// <param name="padding"></param>
        /// <param name="borderMode"></param>
        /// <returns></returns>
        public static int GetFilterGridLength(int inputLength, int filterSize, 
            int stride, int padding, BorderMode borderMode)
        {
            // BorderMode.Same pads with half the filter size on both sides (one less on
            // the second side for an even filter size)
            if (borderMode == BorderMode.Same && filterSize % 2 == 0) 
            {
                return (int)Math.Floor((inputLength + (padding + padding - 1) - filterSize) / (double)stride + 1);
            }

            return (int)Math.Floor((inputLength + padding * 2 - filterSize) / (double)stride + 1);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="depth"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <returns></returns>
        public static float GetValueFromIndex(this Matrix<float> m, int n, int c, int h, int w, 
            int depth, int width, int height)
        {
            var indexInBatchItem = c * width * height + h * width + w;
            var mIndex = indexInBatchItem * m.RowCount + n;
            return m.Data()[mIndex];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="depth"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <returns></returns>
        public static int GetDataIndex(this Matrix<float> m, int n, int c, int h, int w, 
            int depth, int width, int height)
        {
            var indexInBatchItem = c * width * height + h * width + w;
            var mIndex = indexInBatchItem * m.RowCount + n;
            return mIndex;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="data_im"></param>
        /// <param name="channels"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="kernel_h"></param>
        /// <param name="kernel_w"></param>
        /// <param name="pad_h"></param>
        /// <param name="pad_w"></param>
        /// <param name="stride_h"></param>
        /// <param name="stride_w"></param>
        /// <param name="borderMode"></param>
        /// <param name="data_col"></param>
        public static void Batch_Im2Col(Matrix<float> data_im, int channels, int height, int width,
            int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, 
            BorderMode borderMode, Matrix<float> data_col)
        {
            int height_col = GetFilterGridLength(height, kernel_h, stride_h, pad_h, borderMode);
            int width_col = GetFilterGridLength(width, kernel_w, stride_w, pad_w, borderMode);
            var batchSize = data_im.RowCount;

            int channels_col = channels * kernel_h * kernel_w;

            var data_imData = data_im.Data();
            var data_colData = data_col.Data();

            Parallel.For(0, batchSize, batchItem =>
            {
                var batchRowOffSet = batchItem * width_col * height_col;

                for (int c = 0; c < channels_col; ++c)
                {
                    int w_offset = c % kernel_w;
                    int h_offset = (c / kernel_w) % kernel_h;
                    int c_im = c / kernel_h / kernel_w;
                    var cImRowOffSet = c_im * height;

                    for (int h = 0; h < height_col; ++h)
                    {
                        var rowOffSet = h * width_col;
                        for (int w = 0; w < width_col; ++w)
                        {
                            int h_pad = h * stride_h - pad_h + h_offset;
                            int w_pad = w * stride_w - pad_w + w_offset;

                            var outColIndex = batchRowOffSet + rowOffSet + w;
                            var outputIndex = outColIndex * data_col.RowCount + c;                                                   

                            var inputColIndex = (cImRowOffSet + h_pad) * width + w_pad;
                            var inputIndex = inputColIndex * batchSize + batchItem;

                            if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                            {
                                data_colData[outputIndex] = data_imData[inputIndex];
                            }
                            else
                            {
                                data_colData[outputIndex] = 0;
                            }
                        }
                    }
                }
            });
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="convoluted"></param>
        /// <param name="channels"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="kernel_h"></param>
        /// <param name="kernel_w"></param>
        /// <param name="pad_h"></param>
        /// <param name="pad_w"></param>
        /// <param name="stride_h"></param>
        /// <param name="stride_w"></param>
        /// <param name="borderMode"></param>
        /// /// <param name="data_convolutedRowMajor"></param>
        public static void ReshapeConvolutionsToRowMajor(Matrix<float> convoluted, 
            int channels, int height, int width,
            int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, 
            BorderMode borderMode, Matrix<float> data_convolutedRowMajor)
        {
            int height_col = GetFilterGridLength(height, kernel_h, stride_h, pad_h, borderMode);
            int width_col = GetFilterGridLength(width, kernel_w, stride_w, pad_w, borderMode);

            var columnWidth = height_col * width_col;
            var filterCount = convoluted.RowCount;
            var batchSize = data_convolutedRowMajor.RowCount;

            var convolutedData = convoluted.Data();
            var data_convolutedRowMajorData = data_convolutedRowMajor.Data();

            Parallel.For(0, batchSize, batchItem =>
            {
                var batchOffSet = batchItem * columnWidth;

                for (int filter = 0; filter < filterCount; filter++)
                {
                    var rowOffSet = filter * columnWidth;
                    for (int column = 0; column < columnWidth; column++)
                    {
                        // get value from conv data
                        var convIndex = (batchOffSet + column) * filterCount + filter;
                        var convValue = convolutedData[convIndex];

                        // set value in row major data
                        var rowMajorIndex = rowOffSet + column;
                        var index = rowMajorIndex * batchSize + batchItem;
                        data_convolutedRowMajorData[index] = convValue;
                    }
                }
            });
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="data_convolutedRowMajor"></param>
        /// <param name="channels"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="kernel_h"></param>
        /// <param name="kernel_w"></param>
        /// <param name="pad_h"></param>
        /// <param name="pad_w"></param>
        /// <param name="stride_h"></param>
        /// <param name="stride_w"></param>
        /// <param name="borderMode"></param>
        /// <param name="convoluted"></param>
        public static void ReshapeRowMajorToConvolutionLayout(Matrix<float> data_convolutedRowMajor, 
            int channels, int height, int width,
            int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, 
            BorderMode borderMode, Matrix<float> convoluted)
        {
            int height_col = GetFilterGridLength(height, kernel_h, stride_h, pad_h, borderMode);
            int width_col = GetFilterGridLength(width, kernel_w, stride_w, pad_w, borderMode);

            var columnWidth = height_col * width_col;
            var filterCount = convoluted.RowCount;
            var batchSize = data_convolutedRowMajor.RowCount;

            var convolutedData = convoluted.Data();
            var data_convolutedRowMajorData = data_convolutedRowMajor.Data();

            Parallel.For(0, batchSize, batchItem =>
            {
                var batchOffSet = batchItem * columnWidth;

                for (int filter = 0; filter < filterCount; filter++)
                {
                    var rowOffSet = filter * columnWidth;
                    for (int column = 0; column < columnWidth; column++)
                    {
                        // get value from row major data
                        var rowMajorIndex = rowOffSet + column;
                        var index = rowMajorIndex * batchSize + batchItem;
                        var rowValue = data_convolutedRowMajorData[index];

                        // set value in conv data
                        var convIndex = (batchOffSet + column) * filterCount + filter;
                        convolutedData[convIndex] = rowValue;
                    }
                }
            });
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="data_col"></param>
        /// <param name="channels"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="patch_h"></param>
        /// <param name="patch_w"></param>
        /// <param name="pad_h"></param>
        /// <param name="pad_w"></param>
        /// <param name="stride_h"></param>
        /// <param name="stride_w"></param>
        /// <param name="borderMode"></param>
        /// <param name="data_im"></param>
        public static void Batch_Col2Im(Matrix<float> data_col, int channels, int height, int width,
            int patch_h, int patch_w, int pad_h, int pad_w, int stride_h, int stride_w, 
            BorderMode borderMode, Matrix<float> data_im)
        {
            int height_col = GetFilterGridLength(height, patch_h, stride_h, pad_h, borderMode);
            int width_col = GetFilterGridLength(width, patch_w, stride_w, pad_w, borderMode);
            int channels_col = channels * patch_h * patch_w;
            var batchSize = data_im.RowCount;

            var data_colData = data_col.Data();
            var data_imData = data_im.Data();

            Parallel.For(0, batchSize, batchItem =>
            {
                var batchRowOffSet = batchItem * width_col * height_col;

                for (int c = 0; c < channels_col; ++c)
                {
                    int w_offset = c % patch_w;
                    int h_offset = (c / patch_w) % patch_h;
                    int c_im = c / patch_h / patch_w;
                    var c_ImRowOffSet = c_im * height;

                    for (int h = 0; h < height_col; ++h)
                    {
                        var rowOffSet = h * width_col;
                        int h_pad = h * stride_h - pad_h + h_offset;

                        for (int w = 0; w < width_col; ++w)
                        {
                            
                            int w_pad = w * stride_w - pad_w + w_offset;
                            if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                            {
                                var colColIndex = batchRowOffSet + rowOffSet + w;
                                var outIndex = (c_ImRowOffSet + h_pad) * width + w_pad;

                                var imIndex = outIndex * batchSize + batchItem;
                                var colIndex = colColIndex * data_col.RowCount + c;

                                var newValue = data_imData[imIndex] + data_colData[colIndex];
                                data_imData[imIndex] = newValue;
                            }
                        }
                    }
                }
            });
        }
    }
}
