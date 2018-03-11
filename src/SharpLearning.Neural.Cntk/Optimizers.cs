using System.Collections.Generic;
using CNTK;

namespace SharpLearning.Neural.Cntk
{
    /// <summary>
    /// Optimizers for CNTK
    /// </summary>
    public static class Optimizers
    {

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
        /// <param name="l1Reguralization">L1 reguralization term. (Default is 0, so no reguralization)</param>
        /// <param name="l2Reguralization">L2 reguralization term. (Default is 0, so no reguralization)</param>
        /// <param name="unitGain"></param>
        /// <param name="epsilon"></param>
        /// <returns></returns>
        public static Learner Adam(IList<Parameter> parameters, double learningRate = 0.001, 
            double momentum = 0.9, double varianceMomentum = 0.999,              
            double l1Reguralization = 0.0,
            double l2Reguralization = 0.0,
            bool unitGain = true, double epsilon = 1e-08f)
        {
            var learningRatePerSample = new TrainingParameterScheduleDouble(learningRate, 1);
            var momentumRate = new TrainingParameterScheduleDouble(momentum, 1);
            var varianceMomentumRate = new TrainingParameterScheduleDouble(varianceMomentum, 1);

            var options = new AdditionalLearningOptions();
            options.l1RegularizationWeight = l1Reguralization;
            options.l1RegularizationWeight = l2Reguralization;

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
    }
}
