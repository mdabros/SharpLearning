using SharpLearning.Containers;
using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.FeatureCandidateSelectors
{
    public sealed class RandomFeatureCandidateSelector : IFeatureCandidateSelector
    {
        readonly Random m_random;
        readonly List<int> m_indices = new List<int>();
        public RandomFeatureCandidateSelector(int seed)
        {
            m_random = new Random(seed);
        }

        public void Select(int selectCount, int totalCount, List<int> candidates)
        {
            if (selectCount > totalCount) { throw new ArgumentException("select count must be at most total count"); }

            if(m_indices.Count != totalCount)
            {
                m_indices.Capacity = totalCount;
                for (int i = 0; i < totalCount; i++)
                {
                    m_indices.Add(i);
                }

                m_indices.Shuffle(m_random);
            }

            if (selectCount > candidates.Capacity)
            {
                candidates.Capacity = selectCount;
            }

            for (int i = 0; i < selectCount; i++)
            {
                candidates.Add(m_indices[i]);
            }
        }
    }
}
