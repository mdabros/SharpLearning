using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Containers.Extensions;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Diagnostics;

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
        /// The weights outputtet by the layer.
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
        public float[] ColumnMeans;

        /// <summary>
        /// The final column variances used at prediction time.
        /// </summary>
        public float[] ColumnVars;

        Matrix<float> m_delta;

        /// <summary>
        /// The wieghts controlling the linear scaling of the normalization.
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

            int N = input.RowCount; // number of items in mni batch
            int C = Depth;
            int H = Height;
            int W = Width;

            bool is_training = BatchColumnMeans != null;
            double eps = 1e-6;

            Parallel.For(0, C, c =>
            {
                float mean = 0;
                float variance = 0;
                
                if(is_training)
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
                    mean = ColumnMeans[c];
                    variance = ColumnVars[c];
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
                    BatchColumnMeans[c] = mean;
                    BatchcolumnVars[c] = variance;
                }
            });

            return OutputActivations;
        }

        /// <summary>
        /// Gets the gradients of the layer. 
        /// For BatchNormalizationLayer this is the gradients of the linear scale + bias controlling the normaliztion.
        /// </summary>
        /// <returns></returns>
        public WeightsAndBiases GetGradients()
        {
            return new WeightsAndBiases(ScaleGradients, BiasGradients);
        }

        /// <summary>
        /// Gets the parameters of the layer. 
        /// For BatchNormalizationLayer this is the linear scale + bias controlling the normaliztion.
        /// </summary>
        /// <returns></returns>
        public WeightsAndBiases GetParameters()
        {
            return new WeightsAndBiases(Scale, Bias);
        }

        /// <summary>
        /// Adds the parameters and gradients of the layer to the list.
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            var all = new ParametersAndGradients(GetParameters(), GetGradients());
            parametersAndGradients.Add(all);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="inputDepth"></param>
        /// <param name="batchSize"></param>
        /// <param name="random"></param>
        public void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, Random random)
        {
            Width = inputWidth;
            Height = inputHeight;
            Depth = inputDepth;
            var fanOutAndIn = Width * Height * Depth;

            var bound = ActivationInitializationBounds.InitializationBound(ActivationFunc, fanOutAndIn, fanOutAndIn);
            var distribution = new ContinuousUniform(-bound, bound, new Random(random.Next()));

            Scale = Matrix<float>.Build.Random(1, fanOutAndIn, distribution); // scale is done pr. feature or pr. feature map.
            Bias = Vector<float>.Build.Dense(fanOutAndIn, 0.0f);

            BatchColumnMeans = new float[inputDepth];
            BatchcolumnVars = new float[inputDepth];

            ColumnMeans = new float[inputDepth];
            ColumnVars = new float[inputDepth];

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

            // currently the means and vars from last batch is used for test time.
            // this should be changed to mean and vars based on a larger sample
            copy.ColumnMeans = new float[Depth];
            copy.ColumnVars = new float[Depth];
            Array.Copy(BatchColumnMeans, copy.ColumnMeans, BatchColumnMeans.Length);
            Array.Copy(BatchcolumnVars, copy.ColumnVars, BatchcolumnVars.Length);

            copy.OutputActivations = Matrix<float>.Build.Dense(batchSize, fanOut);

            layers.Add(copy);
        }
    }
}
