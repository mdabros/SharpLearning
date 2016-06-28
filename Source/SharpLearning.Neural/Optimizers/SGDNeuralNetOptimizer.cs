using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace SharpLearning.Neural.Optimizers
{
    /// <summary>
    /// Neural net optimizer utilizing the nesterov momentum method for stochastic optimization.
    /// https://en.wikipedia.org/wiki/Gradient_descent
    /// </summary>
    public sealed class SGDNeuralNetOptimizer : INeuralNetOptimizer
    {
        List<Matrix<float>> m_coefs;
        List<Vector<float>> m_intercepts;
        readonly float m_learningRateInit;
        readonly LearningRateSchedule m_learningRateType;

        readonly float m_momentum;
        readonly bool m_nesterov;
        readonly float m_power_t;

        float m_currentLearningRate;
        List<Matrix<float>> m_coefsVelocities;
        List<Vector<float>> m_interceptsVelocities;

        List<Matrix<float>> m_updateCoefs;
        List<Vector<float>> m_updateIntercepts;
        List<Matrix<float>> m_updateCoefsWork;
        List<Vector<float>> m_updateInterceptsWork;

        /// <summary>
        /// Neural net optimizer utilizing the nesterov momentum method for stochastic optimization.
        /// https://en.wikipedia.org/wiki/Gradient_descent
        /// </summary>
        /// <param name="learningRateInit">Initial learning rate. Controls the step size when updating the weights. (Default is 0.001)</param>
        /// <param name="learningRateSchedule">Learning rate Schedule. This specifies the schedule for updating the weights.(Defalt is "Constant").
        /// "Constant" uses the initial learning rate in all iterations.
        /// "InvScale" gradueally decreases the learning rate in each iteration using an inverse scaling
        /// exponent of powert.
        /// "Adaptive" reduces the learning rate by dividing by 5 each time the training loss has not gone down 2 times in a row.</param>
        /// <param name="momentum">Momentum for gradient update. Should be between 0 and 1. (Defualt is 0.9)</param>
        /// <param name="nesterov">Wether to use nesterov momentum. (default is true)</param>
        /// <param name="power_t">The exponent of inverse scaling. This is only used when "InvScale" learning rate types is selected. (Default is 0.5)</param>
        public SGDNeuralNetOptimizer(double learningRateInit, LearningRateSchedule learningRateSchedule,
            double momentum = 0.9, bool nesterov = true, double power_t = 0.5f)
        {
            m_learningRateInit = (float)learningRateInit;
            m_currentLearningRate = (float)learningRateInit;
            m_learningRateType = learningRateSchedule;

            m_momentum = (float)momentum;
            m_nesterov = nesterov;
            m_power_t = (float)power_t;
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

            m_coefsVelocities = new List<Matrix<float>>();
            m_interceptsVelocities = new List<Vector<float>>();
            m_updateCoefs = new List<Matrix<float>>();
            m_updateIntercepts = new List<Vector<float>>();
            m_updateCoefsWork = new List<Matrix<float>>();
            m_updateInterceptsWork = new List<Vector<float>>();

            for (int i = 0; i < coefs.Count; i++)
            {
                m_coefsVelocities.Add(Matrix<float>.Build.Dense(m_coefs[i].RowCount, m_coefs[i].ColumnCount));
                m_interceptsVelocities.Add(Vector<float>.Build.Dense(m_intercepts[i].Count));

                m_updateCoefs.Add(Matrix<float>.Build.Dense(m_coefs[i].RowCount, m_coefs[i].ColumnCount));
                m_updateIntercepts.Add(Vector<float>.Build.Dense(m_intercepts[i].Count));
                m_updateCoefsWork.Add(Matrix<float>.Build.Dense(m_coefs[i].RowCount, m_coefs[i].ColumnCount));
                m_updateInterceptsWork.Add(Vector<float>.Build.Dense(m_intercepts[i].Count));
            }
        }

        /// <summary>
        /// Updates the parameters
        /// </summary>
        /// <param name="coefGrad"></param>
        /// <param name="interceptGrad"></param>
        public void UpdateParameters(List<Matrix<float>> coefGrad, List<Vector<float>> interceptGrad)
        {
            ClearWorkingUpdates();
            Update(coefGrad, interceptGrad);

            for (int i = 0; i < m_coefsVelocities.Count; i++)
            {
                m_updateCoefs[i].CopyTo(m_coefsVelocities[i]);
                m_updateIntercepts[i].CopyTo(m_interceptsVelocities[i]);
            }

            if (m_nesterov)
            {
                ClearWorkingUpdates();
                Update(coefGrad, interceptGrad);

                for (int i = 0; i < m_coefs.Count; i++)
                {
                    m_coefs[i].Add(m_updateCoefs[i], m_coefs[i]);
                    m_intercepts[i].Add(m_updateIntercepts[i], m_intercepts[i]);
                }
            }
            else
            {
                for (int i = 0; i < m_coefs.Count; i++)
                {
                    m_coefs[i].Add(m_coefsVelocities[i], m_coefs[i]);
                    m_intercepts[i].Add(m_interceptsVelocities[i], m_intercepts[i]);
                }
            }
        }

        void Update(List<Matrix<float>> coefGrad, List<Vector<float>> interceptGrad)
        {
            for (int i = 0; i < coefGrad.Count; i++)
            {
                coefGrad[i].Multiply(m_currentLearningRate, m_updateCoefs[i]);
                m_coefsVelocities[i].Multiply(m_momentum, m_updateCoefsWork[i]);
                m_updateCoefsWork[i].Subtract(m_updateCoefs[i], m_updateCoefs[i]);

                m_interceptsVelocities[i].Multiply(m_momentum, m_updateInterceptsWork[i]);
                interceptGrad[i].Multiply(m_currentLearningRate, m_updateIntercepts[i]);
                m_updateInterceptsWork[i].Subtract(m_updateIntercepts[i], m_updateIntercepts[i]);
            }
        }

        void ClearWorkingUpdates()
        {
            // clear updates from last iterations 
            m_updateCoefs.ForEach(c => c.Clear());
            m_updateIntercepts.ForEach(i => i.Clear());
            m_updateCoefsWork.ForEach(c => c.Clear());
            m_updateInterceptsWork.ForEach(i => i.Clear());
        }

        /// <summary>
        /// Complete necesarry updates when an iteration ends
        /// </summary>
        /// <param name="samples"></param>
        public void IterationEnds(int samples)
        {
            if(m_learningRateType == LearningRateSchedule.InvScaling)
            {
                m_currentLearningRate += m_learningRateInit / (float)Math.Pow((float)samples + 1, m_power_t);
            }
        }

        /// <summary>
        /// Stop the optimization if necesarry
        /// </summary>
        /// <returns></returns>
        public bool TriggerStopping()
        {
            if(m_learningRateType == LearningRateSchedule.Adaptive)
            {
                if(m_currentLearningRate > 1e-6)
                {
                    m_currentLearningRate /= 5;
                    Trace.WriteLine("Setting learning rate to: " + m_currentLearningRate);
                    return false;
                }
                else
                {
                    Trace.WriteLine("Learning rate too small. Stopping: ");
                    return true;
                }               
            }
            else
            {
                Trace.WriteLine("Stopping");
                return true;
            }
        }
    }
}
