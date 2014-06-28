using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.FeatureCandidateSelectors
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
        public void Select(int selectCount, int totalCount, List<int> candidates)
        {
            if (selectCount > candidates.Capacity)
            {
                candidates.Capacity = selectCount;
            }
            for (int i = 0; i < selectCount; i++)
            {
                candidates.Add(i);
            }
        }
    }
}
