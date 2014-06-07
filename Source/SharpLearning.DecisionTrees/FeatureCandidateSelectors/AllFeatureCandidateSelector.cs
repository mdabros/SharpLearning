using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Selects all features as candidates
    /// </summary>
    public sealed class AllFeatureCandidateSelector : IFeatureCandidateSelector
    {
        /// <summary>
        /// Selects all features as candidates
        /// </summary>
        /// <param name="count"></param>
        /// <param name="candidates"></param>
        public void Select(int count, List<int> candidates)
        {
            if (count > candidates.Capacity)
            {
                candidates.Capacity = count;
            }
            for (int i = 0; i < count; i++)
            {
                candidates.Add(i);
            }
        }
    }
}
