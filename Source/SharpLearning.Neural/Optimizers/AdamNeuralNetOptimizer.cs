using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace SharpLearning.Neural.Optimizers
{
    /// <summary>
    /// Adam method for stochastic optimization. 
    /// is well suited for problems that are large in terms of data and/or parameters.
    /// The method is also appropriate for non-stationary objectives and problems with
    /// very noisy and/or sparse gradients. https://arxiv.org/pdf/1412.6980.pdf
    /// </summary>
    public sealed class AdamNeuralNetOptimizer : INeuralNetOptimizer
    {
        List<Matrix<float>> m_coefs;
        List<Vector<float>> m_intercepts;

        readonly float m_learningRateInit;
        readonly float m_beta1;
        readonly float m_beta2;
        readonly float m_epsilon;

        int m_t;
        float m_currentLearningRate;

        List<Matrix<float>> m_updateCoefs;
        List<Vector<float>> m_updateIntercepts;

        List<Matrix<float>> m_coefsMs;
        List<Vector<float>> m_interceptsMs;

        List<Matrix<float>> m_coefsVs;
        List<Vector<float>> m_interceptsVs;

        List<Matrix<float>> m_updateCoefsWork;
        List<Vector<float>> m_updateInterceptsWork;

        List<Matrix<float>> m_updateCoefsWork2;
        List<Vector<float>> m_updateInterceptsWork2;


        /// <summary>
        /// All default values are from the original Adam paper
        /// </summary>
        /// <param name="learningRateInit">The initial learning rate used. It controls the step-size in updating the weights</param>
        /// <param name="beta1">Exponential decay rate for estimates of first moment vector, should be in range 0 to 1</param>
        /// <param name="beta2">Exponential decay rate for estimates of second moment vector, should be in range 0 to 1</param>
        /// <param name="epsilon">Value for numerical stabilit</param>
        public AdamNeuralNetOptimizer(double learningRateInit = 0.001, double beta1 = 0.9,
                 double beta2 = 0.999, double epsilon = 1e-8)
        {
            if (beta1 < 0.0 || beta1 > 1.0) { throw new ArgumentException("beta1 must be in range 0 to 1. Value was: " + beta1); }
            if (beta2 < 0.0 || beta2 > 1.0) { throw new ArgumentException("beta2 must be in range 0 to 1. Value was: " + beta2); }

            m_learningRateInit = (float)learningRateInit;
            m_currentLearningRate = (float)learningRateInit;
            m_beta1 = (float)beta1;
            m_beta2 = (float)beta2;
            m_epsilon = (float)epsilon;
        }

        /// <summary>
        /// Complete necesarry updates when an iteration ends
        /// </summary>
        /// <param name="samples"></param>
        public void IterationEnds(int samples)
        {
            // no special behavior
        }

        /// <summary>
        /// Set the parameters for optimization
        /// </summary>
        /// <param name="coefs"></param>
        /// <param name="intercepts"></param>
        public void SetParameters(List<Matrix<float>> coefs, List<Vector<float>> intercepts)
        {
            m_coefs = coefs;
            m_intercepts = intercepts;
            m_t = 0;

            m_updateCoefs = new List<Matrix<float>>();
            m_updateIntercepts =  new List<Vector<float>>();

            m_coefsMs = new List<Matrix<float>>();
            m_interceptsMs = new List<Vector<float>>();

            m_coefsVs = new List<Matrix<float>>();
            m_interceptsVs = new List<Vector<float>>();

            m_updateCoefsWork = new List<Matrix<float>>();
            m_updateInterceptsWork = new List<Vector<float>>();

            m_updateCoefsWork2 = new List<Matrix<float>>();
            m_updateInterceptsWork2 = new List<Vector<float>>();

            for (int i = 0; i < coefs.Count; i++)
            {
                m_updateCoefs.Add(Matrix<float>.Build.Dense(m_coefs[i].RowCount, m_coefs[i].ColumnCount));
                m_updateIntercepts.Add(Vector<float>.Build.Dense(m_intercepts[i].Count));

                m_coefsMs.Add(Matrix<float>.Build.Dense(m_coefs[i].RowCount, m_coefs[i].ColumnCount));
                m_interceptsMs.Add(Vector<float>.Build.Dense(m_intercepts[i].Count));

                m_coefsVs.Add(Matrix<float>.Build.Dense(m_coefs[i].RowCount, m_coefs[i].ColumnCount));
                m_interceptsVs.Add(Vector<float>.Build.Dense(m_intercepts[i].Count));

                m_updateCoefsWork.Add(Matrix<float>.Build.Dense(m_coefs[i].RowCount, m_coefs[i].ColumnCount));
                m_updateInterceptsWork.Add(Vector<float>.Build.Dense(m_intercepts[i].Count));

                m_updateCoefsWork2.Add(Matrix<float>.Build.Dense(m_coefs[i].RowCount, m_coefs[i].ColumnCount));
                m_updateInterceptsWork2.Add(Vector<float>.Build.Dense(m_intercepts[i].Count));
            }
        }

        /// <summary>
        /// Stop the optimization if necesarry
        /// </summary>
        /// <returns></returns>
        public bool TriggerStopping()
        {
            Trace.WriteLine("Stopping");
            return true;
        }

        /// <summary>
        /// Updates the parameters
        /// </summary>
        /// <param name="coefGrad"></param>
        /// <param name="interceptGrad"></param>
        public void UpdateParameters(List<Matrix<float>> coefGrad, List<Vector<float>> interceptGrad)
        {
            m_t++;

            ClearMsVs();

            ClearWorkingUpdates();
            CalculateMs(coefGrad, interceptGrad);

            ClearWorkingUpdates();
            CalculateVs(coefGrad, interceptGrad);

            m_currentLearningRate = (float)(m_learningRateInit * Math.Sqrt(1.0 - Math.Pow(m_beta2, m_t))
                / (1.0 - Math.Pow(m_beta1, m_t)));

            ClearWorkingUpdates();
            Update();

            // update parameters
            for (int i = 0; i < m_coefs.Count; i++)
            {
                m_coefs[i].Add(m_updateCoefs[i], m_coefs[i]);
                m_intercepts[i].Add(m_updateIntercepts[i], m_intercepts[i]);
            }
        }

        void ClearWorkingUpdates()
        {
            // clear updates from last iterations 
            m_updateCoefs.ForEach(c => c.Clear());
            m_updateIntercepts.ForEach(i => i.Clear());
            m_updateCoefsWork.ForEach(c => c.Clear());
            m_updateInterceptsWork.ForEach(i => i.Clear());
            m_updateCoefsWork2.ForEach(c => c.Clear());
            m_updateInterceptsWork2.ForEach(i => i.Clear());

        }

        void ClearMsVs()
        {
            m_coefsMs.ForEach(c => c.Clear());
            m_interceptsMs.ForEach(i => i.Clear());
            m_coefsVs.ForEach(c => c.Clear());
            m_interceptsVs.ForEach(i => i.Clear());
        }

        void CalculateMs(List<Matrix<float>> coefGrad, List<Vector<float>> interceptGrad)
        {
            for (int i = 0; i < coefGrad.Count; i++)
            {
                coefGrad[i].Multiply(1.0f - m_beta1, m_updateCoefs[i]);
                m_coefsMs[i].Multiply(m_beta1, m_updateCoefsWork[i]);
                m_updateCoefsWork[i].Add(m_updateCoefs[i], m_coefsMs[i]);

                interceptGrad[i].Multiply(1.0f - m_beta1, m_updateIntercepts[i]);
                m_interceptsMs[i].Multiply(m_beta1, m_updateInterceptsWork[i]);
                m_updateInterceptsWork[i].Add(m_updateIntercepts[i], m_interceptsMs[i]);
            }
        }

        void CalculateVs(List<Matrix<float>> coefGrad, List<Vector<float>> interceptGrad)
        {
            for (int i = 0; i < coefGrad.Count; i++)
            {
                coefGrad[i].PointwiseMultiply(coefGrad[i], m_updateCoefsWork2[i]);

                m_updateCoefsWork2[i].Multiply(1.0f - m_beta2, m_updateCoefs[i]);
                m_coefsVs[i].Multiply(m_beta2, m_updateCoefsWork[i]);
                m_updateCoefsWork[i].Add(m_updateCoefs[i], m_coefsVs[i]);

                interceptGrad[i].PointwiseMultiply(interceptGrad[i], m_updateInterceptsWork2[i]);

                m_updateInterceptsWork2[i].Multiply(1.0f - m_beta2, m_updateIntercepts[i]);
                m_interceptsVs[i].Multiply(m_beta2, m_updateInterceptsWork[i]);
                m_updateInterceptsWork[i].Add(m_updateIntercepts[i], m_interceptsVs[i]);
            }
        }

        void Update()
        {
            for (int i = 0; i < m_coefsMs.Count; i++)
            {
                m_coefsMs[i].Multiply(-m_currentLearningRate, m_coefsMs[i]);
                m_coefsVs[i].Map(f => (float)Math.Sqrt(f) + m_epsilon, m_coefsVs[i]);
                m_coefsMs[i].PointwiseDivide(m_coefsVs[i], m_updateCoefs[i]);

                m_interceptsMs[i].Multiply(-m_currentLearningRate, m_interceptsMs[i]);
                m_interceptsVs[i].Map(f => (float)Math.Sqrt(f) + m_epsilon, m_interceptsVs[i]);
                m_interceptsMs[i].PointwiseDivide(m_interceptsVs[i], m_updateIntercepts[i]);
            }
        }
    }
}
