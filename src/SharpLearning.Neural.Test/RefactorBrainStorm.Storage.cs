using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Neural.Test.RefactorBranStorm
{
    /// <summary>
    /// 
    /// </summary>
    public class Variable
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="trainable"></param>
        /// <param name="preservable"></param>
        Variable(TensorShape shape, bool trainable = false)
        {
            Shape = shape;
            Trainable = trainable;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="trainable"></param>
        /// <param name="preservable"></param>
        Variable(int[] dimensions, bool trainable = false)
            : this(new TensorShape(dimensions), trainable)
        { }

        /// <summary>
        /// 
        /// </summary>
        public TensorShape Shape { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public bool Trainable { get; }

        /// <summary>
        /// 
        /// </summary>
        public int[] Dimensions { get { return Shape.Dimensions; } }

        /// <summary>
        /// 
        /// </summary>
        public int[] DimensionOffSets { get { return Shape.DimensionOffSets; } }

        /// <summary>
        /// 
        /// </summary>
        public int ElementCount { get { return Shape.ElementCount; } }

        /// <summary>
        /// 
        /// </summary>
        public int Rank { get { return Shape.Rank; } }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public Variable Copy()
        {
            return new Variable(Dimensions.ToArray(), Trainable);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Variable CreateTrainable(params int[] dimensions)
        {
            return new Variable(dimensions, true);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Variable Create(params int[] dimensions)
        {
            return new Variable(dimensions, false);
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public class Data<T>
    {
        /// <summary>
        /// 
        /// </summary>
        public Tensor<T> Tensor { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Tensor<T> Gradient { get; set; }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="variable"></param>
        /// <returns></returns>
        public Tensor<T> GetOrAllocateTensor(Variable variable)
        {
            if (Tensor == null)
            {
                Tensor = new Tensor<T>(variable.Shape, DataLayout.RowMajor);
            }

            return Tensor;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="variable"></param>
        /// <returns></returns>
        public Tensor<T> GetOrAllocateGradient(Variable variable)
        {
            if (Gradient == null)
            {
                Gradient = new Tensor<T>(variable.Shape, DataLayout.RowMajor);
            }

            return Gradient;
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public class NeuralNetStorage
    {
        readonly Dictionary<Variable, Data<float>> m_data;

        /// <summary>
        /// 
        /// </summary>
        public NeuralNetStorage()
        {
            m_data = new Dictionary<Variable, Data<float>>();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameters"></param>
        public List<Data<float>> GetTrainableParameters()
        {
            var parameters = new List<Data<float>>();
            foreach (var data in m_data)
            {
                if (data.Key.Trainable)
                {
                    parameters.Add(data.Value);
                }
            }

            return parameters;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<float> GetTensor(Variable shape)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data<float>();
            }

            return m_data[shape].GetOrAllocateTensor(shape);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<float> GetGradient(Variable shape)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data<float>();
            }

            return m_data[shape].GetOrAllocateGradient(shape);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="distribution"></param>
        public void AssignTensor(Variable shape, Func<float> distribution)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data<float>();
            }

            var tensor = m_data[shape].GetOrAllocateTensor(shape);
            tensor.Map(v => distribution());
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="variable"></param>
        /// <param name="data"></param>
        public void AssignTensor(Variable variable, Tensor<float> data)
        {
            if (!m_data.ContainsKey(variable))
            {
                m_data[variable] = new Data<float>();
            }

            // reshape tensor to update batch size.
            if (variable.Shape != data.Shape) 
            {
                m_data.Remove(variable);
                variable.Shape = data.Shape;

                m_data[variable] = new Data<float>();
                var tensor = m_data[variable].GetOrAllocateTensor(variable);
                data.Data.CopyTo(tensor.Data, 0);
            }
            else
            {
                var tensor = m_data[variable].GetOrAllocateTensor(variable);
                data.Data.CopyTo(tensor.Data, 0);
            }           
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="distribution"></param>
        public void AssignGradient(Variable shape, Func<float> distribution)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data<float>();
            }

            var tensor = m_data[shape].GetOrAllocateGradient(shape);
            tensor.Map(v => distribution());
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="variable"></param>
        /// <param name="data"></param>
        public void AssignGradient(Variable variable, Tensor<float> data)
        {
            if (!m_data.ContainsKey(variable))
            {
                m_data[variable] = new Data<float>();
            }

            var gradient = m_data[variable].GetOrAllocateGradient(variable);

            // reshape tensor to update batch size.
            if (variable.Shape != data.Shape)
            {
                m_data.Remove(variable);
                variable.Shape = data.Shape;

                m_data[variable] = new Data<float>();
                var tensor = m_data[variable].GetOrAllocateGradient(variable);
                data.Data.CopyTo(tensor.Data, 0);
            }
            else
            {
                var tensor = m_data[variable].GetOrAllocateGradient(variable);
                data.Data.CopyTo(tensor.Data, 0);
            }
        }
    }
}
