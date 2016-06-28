using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Neural
{
    /// <summary>
    /// Enum for the learning rate schedule of a neural net learner.
    /// </summary>
    public enum LearningRateSchedule
    {
        /// <summary>
        /// Constant learning rate. Uses the initial learning rate in all iterations 
        /// </summary>
        Constant,

        /// <summary>
        /// InvScaling. Gradually decreases the learning rate in each iteration using an inverse scaling exponent of powert
        /// </summary>
        InvScaling,

        /// <summary>
        /// Adaptive. Reduces the learning rate by dividing by 5 each time the training loss has not gone down 2 times in a row
        /// </summary>
        Adaptive,
    }
}
