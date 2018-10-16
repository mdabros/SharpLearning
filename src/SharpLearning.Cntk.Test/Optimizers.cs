using System;
using System.Collections.Generic;
using CNTK;

namespace SharpLearning.Cntk.Test
{
    public static class Optimizers
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

            return CNTKLib.SGDLearner(AsParameterVector(parameters),
                learningRatePerSample, options);
        }

        public static AdditionalLearningOptions SetAdditionalOptions(double l1Regularization, double l2Regularization)
        {
            var options = new AdditionalLearningOptions();
            options.l1RegularizationWeight = l1Regularization;
            options.l2RegularizationWeight = l2Regularization;
            return options;
        }

        public static ParameterVector AsParameterVector(IList<Parameter> input)
        {
            ParameterVector inputVector = new ParameterVector();
            foreach (var element in input)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }
    }
}
