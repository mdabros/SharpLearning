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
    /// Max pool layer
    /// </summary>
    [Serializable]
    public sealed class MaxPool2DLayer : ILayer
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
        /// 
        /// </summary>
        public int InputHeight;

        /// <summary>
        /// 
        /// </summary>
        public int InputWidth;

        /// <summary>
        /// 
        /// </summary>
        public int InputDepth;

        readonly int m_padWidth;
        readonly int m_padHeight;
        readonly int m_stride;
        readonly int m_poolWidth;
        readonly int m_poolHeight;

        /// <summary>
        /// Switches for determining the position of the max during forward and back propagation.
        /// </summary>
        public int[][] Switchx;

        /// <summary>
        /// Switches for determining the position of the max during forward and back propagation.
        /// </summary>
        public int[][] Switchy;

        /// <summary>
        /// 
        /// </summary>
        public Matrix<float> OutputActivations;

        Matrix<float> m_inputActivations;
        Matrix<float> m_delta;

        /// <summary>
        /// Border mode for the pooling operation.
        /// </summary>
        public BorderMode BorderMode;

        /// <summary>
        /// Max pool layer. 
        /// The max pool layers function is to progressively reduce the spatial size of the representation 
        /// to reduce the amount of parameters and computation in the network. 
        /// The reduction is only done on the width and height. Depth dimension is preserved.
        /// </summary>
        /// <param name="poolWidth">The width of the pool area (default is 2)</param>
        /// <param name="poolHeight">The height of the pool area (default is 2)</param>
        /// <param name="stride">Controls the distance between each neighboring pool areas (default is 2)</param>
        /// <param name="padWidth">Zero padding for the width dimension (default is 0)</param>
        /// <param name="padHeight">Zero padding for the height dimension (default is 0)</param>
        public MaxPool2DLayer(int poolWidth, int poolHeight, int stride, int padWidth, int padHeight)
        {
            if (poolWidth < 1) { throw new ArgumentException("poolWidth is less than 1: " + poolWidth); }
            if (poolHeight < 1) { throw new ArgumentException("poolHeight is less than 1: " + poolHeight); }
            if (padWidth < 0) { throw new ArgumentException("padWidth is less than 0: " + padWidth); }
            if (padHeight < 0) { throw new ArgumentException("padHeight is less than 0: " + padHeight); }
            if (stride < 1) { throw new ArgumentException("stride is less than 0: " + stride); }

            m_poolWidth = poolWidth;
            m_poolHeight = poolHeight;
            m_stride = stride;
            m_padWidth = padWidth;
            m_padHeight = padHeight;
            BorderMode = BorderMode.Undefined;
            ActivationFunc = Activation.Undefined;
        }

        /// <summary>
        /// Max pool layer. 
        /// The max pool layers function is to progressively reduce the spatial size of the representation 
        /// to reduce the amount of parameters and computation in the network. 
        /// The reduction is only done on the width and height. Depth dimension is preserved.
        /// </summary>
        /// <param name="poolWidth">The width of the pool area (default is 2)</param>
        /// <param name="poolHeight">The height of the pool area (default is 2)</param>
        /// <param name="stride">Controls the distance between each neighboring pool areas (default is 2)</param>
        /// <param name="borderMode">Border mode of the max pool operation. 
        /// This will set the width and height padding automatically based on the selected border mode: Valid, Same or Full (default is Valid).</param>
        public MaxPool2DLayer(int poolWidth, int poolHeight, int stride = 2, 
            BorderMode borderMode = BorderMode.Valid)
            : this(poolWidth, poolHeight, stride,
                  ConvUtils.PaddingFromBorderMode(poolWidth, borderMode),
                  ConvUtils.PaddingFromBorderMode(poolHeight, borderMode))
        {
            BorderMode = borderMode;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="delta"></param>
        /// <returns></returns>
        public Matrix<float> Backward(Matrix<float> delta)
        {
            // enumerate each batch item one at a time
            Parallel.For(0, delta.RowCount, i =>
            {
                BackwardSingleItem(delta, m_delta, i);
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

            // enumerate each batch item one at a time
            Parallel.For(0, input.RowCount, i =>
            {
                ForwardSingleItem(input, OutputActivations, i);
            });

            return OutputActivations;
        }

        void ForwardSingleItem(Matrix<float> input, Matrix<float> output, int batchItem)
        {
            var batchSize = input.RowCount;
            var inputData = input.Data();
            var outputData = output.Data();

            for (int depth = 0; depth < InputDepth; ++depth)
            {
                var n = depth * this.Width * this.Height; // a counter for switches
                var inputDepthOffSet = depth * InputHeight * InputWidth;
                var outputDeptOffSet = depth * Height * Width;

                for (int ph = 0; ph < Height; ++ph)
                {
                    var poolRowOffSet = ph * Width;

                    int hstart = ph * m_stride - m_padHeight;
                    int hend = Math.Min(hstart + m_poolHeight, InputHeight);
                    hstart = Math.Max(hstart, 0);

                    for (int pw = 0; pw < Width; ++pw)
                    {
                        int wstart = pw * m_stride - m_padWidth;
                        int wend = Math.Min(wstart + m_poolWidth, InputWidth);
                        wstart = Math.Max(wstart, 0);

                        var currentMax = float.MinValue;
                        int winx = -1, winy = -1;

                        for (int h = hstart; h < hend; ++h)
                        {
                            var rowOffSet = h * InputWidth;
                            for (int w = wstart; w < wend; ++w)
                            {
                                var inputColIndex = rowOffSet + w + inputDepthOffSet;
                                var inputIndex = inputColIndex * batchSize + batchItem;

                                var v = inputData[inputIndex];

                                // perform max pooling and store the index the max location.
                                if (v > currentMax)
                                {
                                    currentMax = v;
                                    winx = w;
                                    winy = h;
                                }
                            }
                        }

                        this.Switchx[batchItem][n] = winx;
                        this.Switchy[batchItem][n] = winy;
                        n++;

                        var outputColIndex = poolRowOffSet + pw + outputDeptOffSet;
                        var outputIndex = outputColIndex * output.RowCount + batchItem;
                        outputData[outputIndex] = currentMax;                      
                    }
                }
            }
        }
        
        void BackwardSingleItem(Matrix<float> inputGradient, Matrix<float> outputGradient, int batchItem)
        {
            var batchSize = inputGradient.RowCount;
            var inputData = inputGradient.Data();
            var outputData = outputGradient.Data();

            var switchx = Switchx[batchItem];
            var switchy = Switchy[batchItem];

            for (var depth = 0; depth < this.Depth; depth++)
            {
                var n = depth * this.Width * this.Height;
                var inputDepthOffSet = depth * InputHeight * InputWidth;
                var outputDeptOffSet = depth * Height * Width;

                var x = -this.m_padWidth;
               // var y = -this.m_padHeight;
                for (var ax = 0; ax < this.Width; x += this.m_stride, ax++)
                {
                    var y = -this.m_padHeight;
                    var axOffSet = ax + outputDeptOffSet;
                    for (var ay = 0; ay < this.Height; y += this.m_stride, ay++)
                    {
                        var inputGradientColIndex = ay * Width + axOffSet;
                        var inputGradientIndex = inputGradientColIndex * batchSize + batchItem;
                        var chainGradient = inputData[inputGradientIndex];

                        var outputGradientColIndex = switchy[n] * InputWidth + switchx[n] + inputDepthOffSet;
                        var outputGradientIndex = outputGradientColIndex * outputGradient.RowCount + batchItem;
                        outputData[outputGradientIndex] = chainGradient;
                        n++;
                    }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            // Pool layer does not have any parameters or gradients.
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
            InputWidth = inputWidth;
            InputHeight = inputHeight;
            InputDepth = inputDepth;

            // computed
            this.Depth = this.InputDepth;

            this.Width = ConvUtils.GetFilterGridLength(InputWidth, m_poolWidth, 
                m_stride, m_padWidth, BorderMode);

            this.Height = ConvUtils.GetFilterGridLength(InputHeight, m_poolHeight, 
                m_stride, m_padHeight, BorderMode);

            // store switches for x,y coordinates for where the max comes from, for each output neuron
            this.Switchx = Enumerable.Range(0, batchSize)
                .Select(v => new int[this.Width * this.Height * this.Depth]).ToArray();

            this.Switchy = Enumerable.Range(0, batchSize)
                .Select(v => new int[this.Width * this.Height * this.Depth]).ToArray();

            var fanIn = InputWidth * InputDepth * InputHeight;
            var fanOut = Depth * Width * Height;

            OutputActivations = Matrix<float>.Build.Dense(batchSize, fanOut);
            m_delta = Matrix<float>.Build.Dense(batchSize, fanIn);
        }

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            var batchSize = 1;
            var copy = new MaxPool2DLayer(m_poolWidth, m_poolHeight, 
                m_stride, m_padWidth, m_padHeight);

            copy.BorderMode = BorderMode;

            copy.InputDepth = InputDepth;
            copy.InputWidth = InputWidth;
            copy.InputHeight = InputHeight;

            copy.Depth = this.Depth;
            copy.Width = this.Width;
            copy.Height = this.Height;

            copy.Switchx = Enumerable.Range(0, batchSize)
                .Select(v => new int[this.Width * this.Height * this.Depth]).ToArray();
            copy.Switchy = Enumerable.Range(0, batchSize)
                .Select(v => new int[this.Width * this.Height * this.Depth]).ToArray();

            var fanOut = Width * Height * Depth;
            copy.OutputActivations = Matrix<float>.Build.Dense(batchSize, fanOut);

            layers.Add(copy);
        }
    }
}
