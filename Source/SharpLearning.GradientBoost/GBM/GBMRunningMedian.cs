using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpLearning.Containers.Extensions;
using System.Collections;

namespace SharpLearning.GradientBoost.GBM
{
    /// <summary>
    /// Calculates the running median
    /// </summary>
    public class GBMRunningMedian
    {
        BinaryHeap<double> m_minHeap = new BinaryHeap<double>();
        BinaryHeap<double> m_maxHeap = new BinaryHeap<double>(new DescendingCompare());

        bool m_initialize = true;

        /// <summary>
        /// Resets the members for a new running median
        /// </summary>
        public void Reset()
        {
            m_initialize = true;
            m_maxHeap.Clear();
            m_minHeap.Clear();
        }

        /// <summary>
        /// Adds a new sample to the running median
        /// </summary>
        /// <param name="sample"></param>
        public void AddSample(double sample)
        {
            if(m_initialize)
            {
                if(m_minHeap.Count == 0 && m_maxHeap.Count == 0)
                {
                    m_minHeap.Insert(sample);
                }
                else
                {
                    var minRoot = m_minHeap.Peek();
                    if(minRoot < sample)
                    {
                        m_minHeap.RemoveRoot();
                        m_minHeap.Insert(sample);
                        m_maxHeap.Insert(minRoot);
                    }
                    else
                    {
                        m_maxHeap.Insert(sample);
                    }
                    m_initialize = false;
                }
            }
            else
            {
                if(sample < m_maxHeap.Peek())
                {
                    m_maxHeap.Insert(sample);
                }
                else
                {
                    m_minHeap.Insert(sample);
                }

                var diff = Math.Abs(m_maxHeap.Count - m_minHeap.Count);

                if(diff > 1)
                {
                    if(m_maxHeap.Count > m_minHeap.Count)
                    {
                        var root = m_maxHeap.RemoveRoot();
                        m_minHeap.Insert(root);
                    }
                    else
                    {
                        var root = m_minHeap.RemoveRoot();
                        m_maxHeap.Insert(root);
                    }
                }
            }

        }

        /// <summary>
        /// Return the current median
        /// </summary>
        /// <returns></returns>
        public double Median()
        {
            if (m_maxHeap.Count == 0 && m_minHeap.Count == 0)
            {
                return 0.0;
            }

            if(m_maxHeap.Count == m_minHeap.Count)
            {
                var max = m_maxHeap.Peek();
                var min = m_minHeap.Peek();
                return (max + min) / 2.0;
            }
            else
            {
                if(m_maxHeap.Count > m_minHeap.Count)
                {
                    return m_maxHeap.Peek();
                }
                else
                {
                    return m_minHeap.Peek();
                }
            }
        }

        /// <summary>
        /// double comparer for descending order 
        /// </summary>
        class DescendingCompare : Comparer<double>
        {
            Comparer<double> m_default = Comparer<double>.Default;

            public override int Compare(double x, double y)
            {
                return -m_default.Compare(x, y);
            }
        }

        /// <summary>
        /// Generic BinaryHeap
        /// </summary>
        /// <typeparam name="T"></typeparam>
        class BinaryHeap<T> : IEnumerable<T>
        {
            private IComparer<T> Comparer;
            private List<T> Items = new List<T>();
            public BinaryHeap()
                : this(Comparer<T>.Default)
            {
            }
            public BinaryHeap(IComparer<T> comp)
            {
                Comparer = comp;
            }
            /// <summary>

            /// Get a count of the number of items in the collection.
            /// </summary>
            public int Count
            {
                get { return Items.Count; }
            }
            /// <summary>
            /// Removes all items from the collection.
            /// </summary>
            public void Clear()
            {
                Items.Clear();
            }
            /// <summary>
            /// Sets the capacity to the actual number of elements in the BinaryHeap,
            /// if that number is less than a threshold value.
            /// </summary>

            /// <remarks>
            /// The current threshold value is 90% (.NET 3.5), but might change in a future release.
            /// </remarks>
            public void TrimExcess()
            {
                Items.TrimExcess();
            }
            /// <summary>
            /// Inserts an item onto the heap.
            /// </summary>
            /// <param name="newItem">The item to be inserted.</param>

            public void Insert(T newItem)
            {
                int i = Count;
                Items.Add(newItem);
                while (i > 0 && Comparer.Compare(Items[(i - 1) / 2], newItem) > 0)
                {
                    Items[i] = Items[(i - 1) / 2];
                    i = (i - 1) / 2;
                }
                Items[i] = newItem;
            }
            /// <summary>
            /// Return the root item from the collection, without removing it.
            /// </summary>
            /// <returns>Returns the item at the root of the heap.</returns>
            public T Peek()
            {
                if (Items.Count == 0)
                {
                    throw new InvalidOperationException("The heap is empty.");
                }
                return Items[0];
            }
            /// <summary>
            /// Removes and returns the root item from the collection.
            /// </summary>
            /// <returns>Returns the item at the root of the heap.</returns>
            public T RemoveRoot()
            {
                if (Items.Count == 0)
                {
                    throw new InvalidOperationException("The heap is empty.");
                }
                // Get the first item
                T rslt = Items[0];
                // Get the last item and bubble it down.
                T tmp = Items[Items.Count - 1];
                Items.RemoveAt(Items.Count - 1);
                if (Items.Count > 0)
                {
                    int i = 0;
                    while (i < Items.Count / 2)
                    {
                        int j = (2 * i) + 1;
                        if ((j < Items.Count - 1) && (Comparer.Compare(Items[j], Items[j + 1]) > 0))
                        {
                            ++j;
                        }
                        if (Comparer.Compare(Items[j], tmp) >= 0)
                        {
                            break;
                        }
                        Items[i] = Items[j];
                        i = j;
                    }
                    Items[i] = tmp;
                }
                return rslt;
            }
            IEnumerator<T> IEnumerable<T>.GetEnumerator()
            {
                foreach (var i in Items)
                {
                    yield return i;
                }
            }
            public IEnumerator GetEnumerator()
            {
                return GetEnumerator();
            }
        }
    }

}
