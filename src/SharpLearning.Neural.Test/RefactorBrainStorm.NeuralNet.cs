using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpLearning.Neural.Test.RefactorBranStorm
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class NeuralNet
    {
        readonly List<ILayer> Layers;
        readonly Initialization m_initialization;

        /// <summary>
        /// 
        /// </summary>
        public NeuralNet(Initialization initialization = Initialization.GlorotUniform)
        {
            m_initialization = initialization;
            Layers = new List<ILayer>();
        }

        /// <summary>
        /// 
        /// </summary>
        public Variable Input { get { return Layers.First().Input; } private set { } }

        /// <summary>
        /// 
        /// </summary>
        public Variable Output { get { return Layers.Last().Output; } }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="layer"></param>
        public void Add(ILayer layer)
        {
            if (Layers.Count == 0)
            {
                Input = layer.Input;
            }

            Layers.Add(layer);
        }

        public void Forward(NeuralNetStorage storage)
        {
            Layers.ForEach(l => l.Forward(storage));
        }

        /// <summary>
        /// 
        /// </summary>
        public void Backward(NeuralNetStorage storage)
        {
            for (int i = Layers.Count; i-- > 0;)
            {
                Layers[i].Backward(storage);
            }
        }

        /// <summary>
        /// Initializes the layers in the neural net (Instantiates members and creates random initialization of weights). 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="random"></param>
        public void Initialize(Random random, int batcSize,
            NeuralNetStorage storage)
        {
            if (!(Layers.First() is InputLayer))
            {
                throw new ArgumentException("First layer must be InputLayer. Was: " + Layers.First().GetType().Name);
            }

            Layers.First().Initialize(Input, batcSize, 
                storage, random, m_initialization);

            for (int i = 1; i < Layers.Count; i++)
            {
                var previousLayer = Layers[i - 1];
                Layers[i].Initialize(previousLayer.Output, batcSize, 
                    storage, random, m_initialization);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(int batcSize)
        {
            Layers.First().UpdateDimensions(batcSize);

            for (int i = 1; i < Layers.Count; i++)
            {
                var previousLayer = Layers[i - 1];
                Layers[i].UpdateDimensions(batcSize);
            }
        }

        /// <summary>
        /// Model summary of all shapes and parameters in the network.
        /// </summary>
        /// <returns></returns>
        public string Summary()
        {
            var sb = new StringBuilder();
            sb.AppendLine("---------------------");
            sb.AppendLine("Model Summary");
            sb.AppendLine("Input Shape= " + Input.Shape.ToString());
            sb.AppendLine("Output Shape= " + Output.Shape.ToString());
            sb.AppendLine("=====================");
            sb.AppendLine("");

            var totalParameterCount = 0;

            foreach (var layer in Layers)
            {
                var outputShape = layer.Output.Shape;
                var layerParameterCount = layer.ParameterCount();

                sb.AppendLine($"Layer Name='{layer.GetType().Name}' Output Shape={outputShape.ToString(),-30}" +
                    $" Param #:{layerParameterCount}");

                totalParameterCount += layerParameterCount;
            }

            sb.AppendLine();
            sb.AppendLine("=====================");
            sb.AppendLine($"Total Number of Parameters: {totalParameterCount:N0}");
            sb.AppendLine("---------------------");

            return sb.ToString();
        }
    }
}
