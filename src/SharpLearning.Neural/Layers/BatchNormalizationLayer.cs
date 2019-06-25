using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// BatchNormalizationLayer. Batch normalization can be added to accelerate the learning process of a neural net.
    /// https://arxiv.org/abs/1502.03167
    /// </summary>
    [Serializable]
    public sealed class BatchNormalizationLayer : ILayer
    {
        /// <summary>
        /// 
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public int Height { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public int Depth { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Activation ActivationFunc { get; set; }

        /// <summary>
        /// The weights outputted by the layer.
        /// </summary>
        public Matrix<float> OutputActivations;
        Matrix<float> m_inputActivations;

        /// <summary>
        /// The batch column means.
        /// </summary>
        public float[] BatchColumnMeans;

        /// <summary>
        /// The batch column variances.
        /// </summary>
        public float[] BatchcolumnVars;

        /// <summary>
        /// The final column means used at prediction time.
        /// </summary>
        public float[] MovingAverageMeans;

        /// <summary>
        /// The final column variances used at prediction time.
        /// </summary>
        public float[] MovingAverageVariance;

        Matrix<float> m_delta;

        /// <summary>
        /// The weights controlling the linear scaling of the normalization.
        /// </summary>
        public Matrix<float> Scale;

        /// <summary>
        /// The bias controlling the linear scaling of the normalization.
        /// </summary>
        public Vector<float> Bias;

        Matrix<float> ScaleGradients;
        Vector<float> BiasGradients;

        /// <summary>
        /// BatchNormalizationLayer. Batch normalization can be added to accelerate the learning process of a neural net.
        /// https://arxiv.org/abs/1502.03167
        /// </summary>
        public BatchNormalizationLayer()
        {
            ActivationFunc = Activation.Undefined;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="delta"></param>
        /// <returns></returns>
        public Matrix<float> Backward(Matrix<float> delta)
        {
            var src = m_inputActivations;
            var diff_dst = delta;
            var scaleshift = Scale;
            var diff_src = m_delta;
            var diff_scaleshift = ScaleGradients;

            int N = diff_src.RowCount;
            int C = Depth;
            int H = Height;
            int W = Width;

            Parallel.For(0, C, c =>
            {
                var mean = BatchColumnMeans[c];
                var variance = BatchcolumnVars[c];
                var gamma = scaleshift.At(0, c);
                var diff_gamma = 0.0f;
                var diff_beta = 0.0f;

                for (int n = 0; n < N; ++n)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                        {
                            diff_gamma += (src.GetValueFromIndex(n, c, h, w, Depth, Width, Height) - mean)
                                * diff_dst.GetValueFromIndex(n, c, h, w, Depth, Width, Height);
                            diff_beta += diff_dst.GetValueFromIndex(n, c, h, w, Depth, Width, Height);
                        }
                diff_gamma *= variance;

                ScaleGradients.At(0, c, diff_gamma);
                BiasGradients[c] = diff_beta;

                for (int n = 0; n < N; ++n)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                        {
                            var diffSrcIndex = diff_src.GetDataIndex(n, c, h, w, Depth, Width, Height);
                            diff_src.Data()[diffSrcIndex] =
                                diff_dst.Data()[diffSrcIndex] - diff_beta / (W * H * N)
                                - (src.Data()[diffSrcIndex] - mean)
                                * diff_gamma * variance / (W * H * N);
                            diff_src.Data()[diffSrcIndex] *= gamma * variance;
                        }
            });

            return m_delta;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix<float> Forward(Matrix<float> input)
        {
            m_inputActivations = input;

            var src = input;
            var dst = OutputActivations.Data();

            int N = input.RowCount; // number of items in mini batch
            int C = Depth;
            int H = Height;
            int W = Width;

            bool is_training = BatchColumnMeans != null;
            double eps = 1e-6;

            Parallel.For(0, C, c =>
            {
                float mean = 0;
                float variance = 0;

                if (is_training)
                {
                    for (int n = 0; n < N; ++n)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w)
                                mean += src.GetValueFromIndex(n, c, h, w, Depth, Width, Height);
                    mean /= W * N * H;

                    for (int n = 0; n < N; ++n)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w)
                            {
                                var m = src.GetValueFromIndex(n, c, h, w, Depth, Width, Height) - mean;
                                variance += m * m;
                            }
                    variance = 1f / (float)Math.Sqrt(variance / (W * H * N) + eps);
                }
                else
                {
                    mean = MovingAverageMeans[c];
                    variance = MovingAverageVariance[c];
                }

                for (int n = 0; n < N; ++n)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w)
                        {
                            var d_off = src.GetDataIndex(n, c, h, w, Depth, Width, Height);
                            var scale = Scale.At(0, c);
                            var bias = Bias[c];
                            dst[d_off] = scale * (src.Data()[d_off] - mean) * variance + bias;
                        }

                if (is_training)
                {
                    MovingAverageMeans[c] = MovingAverage(MovingAverageMeans[c], mean);
                    MovingAverageVariance[c] = MovingAverage(MovingAverageVariance[c], variance);

                    BatchColumnMeans[c] = mean;
                    BatchcolumnVars[c] = variance;
                }
            });

            return OutputActivations;
        }

        float MovingAverage(float currentValue, float value, float momentum = 0.99f)
        {
            var newValue = currentValue * momentum + value * (1.0f - momentum);
            return newValue;
        }

        /// <summary>
        /// Adds the parameters and gradients of the layer to the list.
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            var scale = new ParametersAndGradients(Scale.Data(), ScaleGradients.Data());
            var bias = new ParametersAndGradients(Bias.Data(), BiasGradients.Data());

            parametersAndGradients.Add(scale);
            parametersAndGradients.Add(bias);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="inputDepth"></param>
        /// <param name="batchSize"></param>
        /// <param name="initializtion"></param>
        /// <param name="random"></param>
        public void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, 
            Initialization initializtion, Random random)
        {
            Width = inputWidth;
            Height = inputHeight;
            Depth = inputDepth;
            var fanOutAndIn = Width * Height * Depth;

            Scale = Matrix<float>.Build.Dense(1, fanOutAndIn, 1.0f); // scale is done pr. feature or pr. feature map.
            Bias = Vector<float>.Build.Dense(fanOutAndIn, 0.0f);

            BatchColumnMeans = new float[inputDepth];
            BatchcolumnVars = new float[inputDepth];

            MovingAverageMeans = new float[inputDepth];
            MovingAverageVariance = Enumerable.Range(0, inputDepth).Select(v => 1.0f).ToArray();

            ScaleGradients = Matrix<float>.Build.Dense(1, fanOutAndIn, 1);
            BiasGradients = Vector<float>.Build.Dense(fanOutAndIn);

            OutputActivations = Matrix<float>.Build.Dense(batchSize, fanOutAndIn);

            m_delta = Matrix<float>.Build.Dense(batchSize, fanOutAndIn);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            var batchSize = 1; // prediction time only uses 1 item at a time.
            var fanOut = Width * Height * Depth;
            var copy = new BatchNormalizationLayer();

            copy.Width = Width;
            copy.Height = Height;
            copy.Depth = Depth;

            copy.Scale = Matrix<float>.Build.Dense(Scale.RowCount, Scale.ColumnCount);
            copy.Bias = Vector<float>.Build.Dense(Bias.Count);
            Array.Copy(Scale.Data(), copy.Scale.Data(), Scale.Data().Length);
            Array.Copy(Bias.Data(), copy.Bias.Data(), Bias.Data().Length);

            copy.MovingAverageMeans = new float[Depth];
            copy.MovingAverageVariance = new float[Depth];
            Array.Copy(MovingAverageMeans, copy.MovingAverageMeans, MovingAverageMeans.Length);
            Array.Copy(MovingAverageVariance, copy.MovingAverageVariance, MovingAverageVariance.Length);

            copy.OutputActivations = Matrix<float>.Build.Dense(batchSize, fanOut);

            layers.Add(copy);
        }
    }
}
