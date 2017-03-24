using System;
using System.Linq;
using System.Collections.Generic;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public class NeuralNetStorage
    {
        readonly Dictionary<Variable, Data<double>> m_data;

        /// <summary>
        /// 
        /// </summary>
        public NeuralNetStorage()
        {
            m_data = new Dictionary<Variable, Data<double>>();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameters"></param>
        public void GetTrainableParameters(List<Data<double>> parameters)
        {
            foreach (var data in m_data)
            {
                if(data.Key.Trainable)
                {
                    parameters.Add(data.Value);
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameters"></param>
        public void ClearNonTrainables()
        {
            var variablesToClear = m_data.Keys.Where(k => !k.Trainable)
                .ToList();

            foreach (var key in variablesToClear)
                m_data.Remove(key);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<double> GetTensor(Variable shape)
        {
            if(!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data<double>();
            }

            return m_data[shape].GetOrAllocateTensor(shape);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<double> GetGradient(Variable shape)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data<double>();
            }

            return m_data[shape].GetOrAllocateGradient(shape);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="distribution"></param>
        public void AssignTensor(Variable shape, Func<double> distribution)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data<double>();
            }

            var tensor = m_data[shape].GetOrAllocateTensor(shape);
            tensor.Map(v => distribution());
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="data"></param>
        public void AssignTensor(Variable shape, double[] data)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data<double>();
            }

            var tensor = m_data[shape].GetOrAllocateTensor(shape);
            
            // consider reshape instead of fail.

            if(tensor.ElementCount != data.Length)
            {
                throw new ArgumentException($"tensor element count: {tensor.ElementCount}, differs from data length: {data.Length}");
            }

            data.CopyTo(tensor.Data, 0);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="distribution"></param>
        public void AssignGradient(Variable shape, Func<double> distribution)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data<double>();
            }

            var tensor = m_data[shape].GetOrAllocateGradient(shape);
            tensor.Map(v => distribution());
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="data"></param>
        public void AssignGradient(Variable shape, double[] data)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data<double>();
            }

            var gradient = m_data[shape].GetOrAllocateGradient(shape);

            // consider reshape instead of fail.

            if (gradient.ElementCount != data.Length)
            {
                throw new ArgumentException($"tensor element count: {gradient.ElementCount}, differs from data length: {data.Length}");
            }

            data.CopyTo(gradient.Data, 0);
        }
    }
}
