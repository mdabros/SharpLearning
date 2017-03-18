using System;
using System.Collections.Generic;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public class Executor
    {
        readonly Dictionary<Variable, Data> m_data;

        /// <summary>
        /// 
        /// </summary>
        public Executor()
        {
            m_data = new Dictionary<Variable, Data>();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<float> GetTensor(Variable shape)
        {
            if(!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data();
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
                m_data[shape] = new Data();
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
                m_data[shape] = new Data();
            }

            var tensor = m_data[shape].GetOrAllocateTensor(shape);
            tensor.Map(v => distribution());
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="data"></param>
        public void AssignTensor(Variable shape, float[] data)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data();
            }

            var tensor = m_data[shape].GetOrAllocateTensor(shape);
            
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
        public void AssignGradient(Variable shape, Func<float> distribution)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data();
            }

            var tensor = m_data[shape].GetOrAllocateGradient(shape);
            tensor.Map(v => distribution());
        }
    }
}
