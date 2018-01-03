using System;
using System.Collections.Generic;
using CNTK;

namespace SharpLearning.Neural.Cntk
{
    public static class CntkUtils
    {
        public static void ConvertArray(double[] a1, float[] a2)
        {
            if (a1.Length != a2.Length)
            { throw new ArgumentException("a1 length: " + a1.Length + " differs from " + a2.Length); }

            for (int i = 0; i < a1.Length; i++)
            {
                a2[i] = (float)a1[i];
            }
        }

        /// <summary>
        /// create a Adam learner
        /// </summary>
        /// <param name="parameters">parameters to learn</param>
        /// <param name="learningRateSchedule">learning rate schedule</param>
        /// <param name="momentumSchedule">momentum schedule</param>
        /// <param name="varianceMomentumSchedule">momentum schedule</param>
        /// <param name="epsilon"></param>
        /// <param name="adamax"></param>
        /// <param name="unitGain">unit gain</param>
        /// <param name="additionalOptions"></param>
        /// <returns></returns>
        public static Learner AdamLearner(IList<Parameter> parameters, TrainingParameterScheduleDouble learningRateSchedule,
            TrainingParameterScheduleDouble momentumSchedule, TrainingParameterScheduleDouble varianceMomentumSchedule,
            double epsilon = 1e-8, bool adamax = false,
            bool unitGain = false, AdditionalLearningOptions additionalOptions = null)
        {
            if (additionalOptions == null)
            {
                additionalOptions = new AdditionalLearningOptions();
            }

            ParameterVector parameterVector = AsParameterVector(parameters);
            return CNTKLib.AdamLearner(parameterVector, learningRateSchedule, momentumSchedule, unitGain,
                varianceMomentumSchedule, epsilon, adamax, additionalOptions);
        }

        /// <summary>
        /// create a Adam learner
        /// </summary>
        /// <param name="parameters">parameters to learn</param>
        /// <param name="epsilon"></param>
        /// <param name="adamax"></param>
        /// <param name="unitGain">unit gain</param>
        /// <param name="additionalOptions"></param>
        /// <returns></returns>
        public static Learner AdamLearner(IList<Parameter> parameters,
            double epsilon = 1e-8, bool adamax = false,
            bool unitGain = false, AdditionalLearningOptions additionalOptions = null)
        {
            return AdamLearner(parameters,
                new TrainingParameterScheduleDouble(0.001, 1),
                new TrainingParameterScheduleDouble(0.9, 1),
                new TrainingParameterScheduleDouble(0.999, 1),
                epsilon, adamax, unitGain,
                additionalOptions);
        }

        internal static ParameterVector AsParameterVector(IList<Parameter> input)
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
