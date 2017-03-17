using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpLearning.Neural.Activations;

namespace SharpLearning.Neural.Descriptors
{
    /// <summary>
    /// 
    /// </summary>
    public class Activation
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly NonLinearity NonLinearity;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="nonLinearity"></param>
        public Activation(NonLinearity nonLinearity)
        {
            NonLinearity = nonLinearity;
        }
    }
}
