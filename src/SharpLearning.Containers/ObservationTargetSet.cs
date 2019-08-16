using System;
using System.Linq;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers
{
    /// <summary>
    /// Container for storing an observations and targets pair.
    /// </summary>
    public sealed class ObservationTargetSet : IEquatable<ObservationTargetSet>
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly F64Matrix Observations;

        /// <summary>
        /// 
        /// </summary>
        public readonly double[] Targets;

        /// <summary>
        /// Container for storing an observations and targets pair.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        public ObservationTargetSet(F64Matrix observations, double[] targets)
	    {
            Observations = observations ?? throw new ArgumentNullException(nameof(observations));
            Targets = targets ?? throw new ArgumentNullException(nameof(targets));
	    }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(ObservationTargetSet other)
        {
            if (!Observations.Equals(other.Observations)) { return false; }
            if (!Targets.SequenceEqual(other.Targets)) { return false; }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj is ObservationTargetSet other &&  this.Equals(other))
            {
                return true;
            }

            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;
                hash = hash * 23 + Observations.GetHashCode();
                hash = hash * 23 + Targets.GetHashCode();

                return hash;
            }
        }
    }
}
