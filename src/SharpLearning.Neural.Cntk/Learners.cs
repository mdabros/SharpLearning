using System;
using System.Collections.Generic;
using CNTK;

namespace SharpLearning.Neural.Cntk
{
    /// <summary>
    /// Learners for CNTK
    /// </summary>
    public static class Learners
    {
        /// <summary>
        /// Stochastic Gradient Descent.
        /// </summary>
        /// <param name="parameters">Learnable parameters of the model</param>
        /// <param name="learningRate">Learning rate. (Default is 0.001)</param>
        /// <param name="l1Regularization">L1 regularization term. (Default is 0, so no regularization)</param>
        /// <param name="l2Regularization">L2 regularization term. (Default is 0, so no regularization)</param>
        /// <returns></returns>
        public static Learner SGD(IList<Parameter> parameters,
            double learningRate = 0.01,
            double l1Regularization = 0.0,
            double l2Regularization = 0.0)
        {
            if (parameters == null) { throw new ArgumentNullException("parameters"); }
            if (learningRate < 0.0) { throw new ArgumentException(nameof(learningRate) + " Has to be larger or equal to 0"); }
            if (l1Regularization < 0.0) { throw new ArgumentException(nameof(l1Regularization) + " Has to be larger or equal to 0"); }
            if (l2Regularization < 0.0) { throw new ArgumentException(nameof(l2Regularization) + " Has to be larger or equal to 0"); }

            var learningRatePerSample = new TrainingParameterScheduleDouble(learningRate, 1);

            var options = SetAdditionalOptions(l1Regularization, l2Regularization);

            return CNTKLib.SGDLearner(CntkUtils.AsParameterVector(parameters),
                learningRatePerSample, options);
        }

        /// <summary>
        /// Stochastic Gradient Descent with momentum.
        /// </summary>
        /// <param name="parameters">Learnable parameters of the model</param>
        /// <param name="learningRate">Learning rate. (Default is 0.001)</param>
        /// <param name="momentum">Momentum (default is 0.9).</param>
        /// <param name="l1Regularization">L1 regularization term. (Default is 0, so no regularization)</param>
        /// <param name="l2Regularization">L2 regularization term. (Default is 0, so no regularization)</param>
        /// <param name="unitGain"></param>
        /// <returns></returns>
        public static Learner MomentumSGD(IList<Parameter> parameters, 
            double learningRate = 0.01, double momentum = 0.1,
            double l1Regularization = 0.0,
            double l2Regularization = 0.0,
            bool unitGain = true)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(learningRate, 1);
            var momentumPerSample = new TrainingParameterScheduleDouble(momentum, 1);

            var options = SetAdditionalOptions(l1Regularization, l2Regularization);

            return CNTKLib.MomentumSGDLearner(CntkUtils.AsParameterVector(parameters), 
                learningRatePerSample, momentumPerSample, unitGain,
                options);
        }

        /// <summary>
        /// Adam (Adaptive Moment Estimation) is another method that computes adaptive learning rates for each parameter. 
        /// In addition to storing an exponentially decaying average of past squared gradients vtvt like Adadelta, 
        /// Adam also keeps an exponentially decaying average of past gradients, similar to momentum.
        /// Essentially Adam is RMSProp with momentum.
        /// https://arxiv.org/pdf/1412.6980.pdf.
        /// </summary>
        /// <param name="parameters">Learnable parameters of the model</param>
        /// <param name="learningRate">Learning rate. (Default is 0.001)</param>
        /// <param name="momentum">Momentum (default is 0.9).
        /// Note that this is the beta1 parameter in the Adam paper.</param>
        /// <param name="varianceMomentum">variance momentum schedule. (Default is 0.999). 
        /// Note that this is the beta2 parameter in the Adam paper.</param>
        /// <param name="l1Regularization">L1 regularization term. (Default is 0, so no regularization)</param>
        /// <param name="l2Regularization">L2 regularization term. (Default is 0, so no regularization)</param>
        /// <param name="unitGain"></param>
        /// <param name="epsilon"></param>
        /// <returns></returns>
        public static Learner Adam(IList<Parameter> parameters, double learningRate = 0.001, 
            double momentum = 0.9, double varianceMomentum = 0.999,              
            double l1Regularization = 0.0,
            double l2Regularization = 0.0,
            bool unitGain = true, double epsilon = 1e-08f)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(learningRate, 1);
            var momentumRate = new TrainingParameterScheduleDouble(momentum, 1);
            var varianceMomentumRate = new TrainingParameterScheduleDouble(varianceMomentum, 1);

            var options = SetAdditionalOptions(l1Regularization, l2Regularization);

            // Consider: gradientClippingWithTruncation and gradientClippingThresholdPerSample

            return CNTKLib.AdamLearner(CntkUtils.AsParameterVector(parameters),
                learningRatePerSample,
                momentumRate,
                unitGain,
                varianceMomentumRate,
                epsilon,
                false,
                options);
        }

        public static AdditionalLearningOptions SetAdditionalOptions(double l1Regularization, double l2Regularization)
        {
            var options = new AdditionalLearningOptions();
            options.l1RegularizationWeight = l1Regularization;
            options.l2RegularizationWeight = l2Regularization;
            return options;
        }
    }
}
