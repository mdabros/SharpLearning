using System;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    /// <summary>
    /// Maintains weighted target counts. 
    /// Offset is used for cases with negative target names like -1.
    /// This is alot faster than mapping using a dictionary since this solution simply indexes into an array
    /// </summary>
    internal class TargetCounts
    {
        public double[] Counts;
        public int OffSet;
        public int Length;

        public TargetCounts()
            : this(0, 0)
        {
	    }

        public TargetCounts (int size, int offset)
	    {
            OffSet = offset;
            Length = offset + size;
            Counts = new double[Length];
	    }

        public double this[int index]
        {
            get { return Counts[OffSet + index]; }
            set { Counts[OffSet + index] = value; }
        }

        /// <summary>
        /// Clears the counts
        /// </summary>
        public void Clear()
        {
            Array.Clear(Counts, 0, Counts.Length);
        }

        /// <summary>
        /// Resets the size and off sets and clears 
        /// the counts  
        /// </summary>
        /// <param name="size"></param>
        /// <param name="offset"></param>
        public void Reset(int size, int offset)
        {
            OffSet = offset;
            Length = offset + size;

            Array.Resize(ref Counts, Length);
            Array.Clear(Counts, 0, Counts.Length);
        }

        /// <summary>
        /// Sets the counts equal to the prvided counts.
        /// </summary>
        /// <param name="newCounts"></param>
        public void SetCounts(TargetCounts newCounts)
        {
            Array.Copy(newCounts.Counts, Counts, newCounts.Counts.Length);
        }
    }
}
