using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using System;
using System.Linq;
using System.Collections.Generic;

namespace SharpLearning.Neural
{
    /// <summary>
    /// Neural net consisting of a set of layers.
    /// </summary>
    [Serializable]
    public sealed class NeuralNet
    {
        /// <summary>
        /// The layers in the network
        /// </summary>
        public readonly List<ILayer> Layers;

        /// <summary>
        /// 
        /// </summary>
        public NeuralNet()
        {
            Layers = new List<ILayer>();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="layers"></param>
        private NeuralNet(List<ILayer> layers)
        {
            if(layers == null)
            {
                throw new ArgumentNullException("layers");
            }

            Layers = layers;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="layer"></param>
        public void Add(ILayer layer)
        {
            var unitsOfPreviousLayer = 0;
            if(Layers.Count > 0)
            {
                unitsOfPreviousLayer = Layers[Layers.Count - 1].Width;
            }

            if(layer is IOutputLayer)
            {
                var denseLayer = new DenseLayer(layer.Depth, Activation.Undefined);
                Layers.Add(denseLayer);
            }

            Layers.Add(layer);

            if(layer is IBatchNormalizable) // consider adding seperate interface for batch normalization
            {
                if(((IBatchNormalizable)layer).UseBatchNormalization)
                {
                    Layers.Add(new BatchNormalizationLayer());
                }
            }

            if(layer.ActivationFunc != Activation.Undefined)
            {
                Layers.Add(new ActivationLayer(layer.ActivationFunc));
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="delta"></param>
        public void Backward(Matrix<float> delta)
        {
            for (int i = Layers.Count; i-- > 0;)
            {
                delta = Layers[i].Backward(delta);                
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix<float> Forward(Matrix<float> input)
        {
            var activation = Layers[0].Forward(input);
            for (int i = 1; i < Layers.Count; i++)
            {
                activation = Layers[i].Forward(activation);
            }

            return activation;
        }


        /// <summary>
        /// Forwards each observations from input and stores the results in output.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        public void Forward(Matrix<float> input, Matrix<float> output)
        {
            var row = Matrix<float>.Build.Dense(1, input.ColumnCount);
            for (int rowIndex = 0; rowIndex < input.RowCount; rowIndex++)
            {
                input.Row(rowIndex, row.Data());
                var prediction = Forward(row);

                for (int col = 0; col < prediction.ColumnCount; col++)
                {
                    output[rowIndex, col] = prediction[0, col];
                }
            }
        }

        /// <summary>
        /// Initializes the layers in the neural net (Instantiates members and creates random initialization of weights). 
        /// </summary>
        /// <param name="batchSize"></param>
        /// <param name="random"></param>
        public void Initialize(int batchSize, Random random)
        {
            if (!(Layers.First() is InputLayer))
            {
                throw new ArgumentException("First layer must be InputLayer. Was: " + Layers.First().GetType().Name);
            }

            if(!(Layers.Last() is IOutputLayer))
            {
                throw new ArgumentException("Last layer must be an output layer type. Was: " + Layers.Last().GetType().Name);
            }

            for (int i = 1; i < Layers.Count; i++)
            {
                var previousLayer = Layers[i - 1];
                Layers[i].Initialize(previousLayer.Width, previousLayer.Height, previousLayer.Depth, batchSize, random);
            }
        }

        /// <summary>
        /// Returns all paramters and gradients from the net.
        /// </summary>
        public List<ParametersAndGradients> GetParametersAndGradients()
        {
            var paramtersAndGradients = new List<ParametersAndGradients>();
            Layers.ForEach(l => l.AddParameresAndGradients(paramtersAndGradients));
            return paramtersAndGradients;
        }

        /// <summary>
        /// Copies a minimal version of the neural net to be used in a model for predictions.
        /// </summary>
        public NeuralNet CopyNetForPredictionModel()
        {
            var layers = new List<ILayer>();
            foreach (var layer in Layers)
            {
                layer.CopyLayerForPredictionModel(layers);
            }

            return NeuralNet.CreateNetFromInitializedLayers(layers);
        }

        /// <summary>
        /// Creates a neural net from already initialized layers. 
        /// This means that layer.Initialize will not be called.
        /// </summary>
        /// <param name="layers"></param>
        /// <returns></returns>
        public static NeuralNet CreateNetFromInitializedLayers(List<ILayer> layers)
        {
            return new NeuralNet(layers);
        }
    }
}
