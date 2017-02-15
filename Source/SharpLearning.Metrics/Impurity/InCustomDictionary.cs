using System;
using System.Collections;
using System.Collections.Generic;

namespace SharpLearning.Metrics.Impurity
{
    // http://blog.teamleadnet.com/2012/07/ultra-fast-hashtable-dictionary-with.html
    // https://code.google.com/p/mapreduce-net/source/browse/MapReduce.NET/Collections/CustomDictionary.cs
    internal class IntCustomDictionary : IDictionary<int, int>, IDictionary
    {
        static readonly uint[] m_primeSizes = new uint[]{ 89, 179, 359, 719, 1439, 2879, 5779, 11579, 23159, 46327,
                                                          92657, 185323, 370661, 741337, 1482707, 2965421, 5930887, 11861791,
                                                          23723599, 47447201, 94894427, 189788857, 379577741, 759155483};

        int[] m_hashes;
        DictionaryEntry[] m_entries;
        int m_nextfree = 0;

        internal struct DictionaryEntry
        {
            public int Key;
            public int Next;
            public int Value;
            public uint Hashcode;
        }

        public IntCustomDictionary()
        {
            var initialSize = m_primeSizes[0];
            this.m_hashes = new int[initialSize];
            this.m_entries = new DictionaryEntry[initialSize];
            ClearHashes();
        }

        public int InitOrGetPosition(int key)
        {
            return AddNew(key, default(int));
        }

        public int GetAtPosition(int pos)
        {
            return m_entries[pos].Value;
        }

        public void StoreAtPosition(int pos, int value)
        {
            m_entries[pos].Value = value;
        }

        public int AddNew(int key, int value)
        {
            if (m_nextfree >= m_entries.Length)
                Resize();

            uint hash = (uint)key;

            uint hashPos = hash % (uint)m_hashes.Length;

            int entryLocation = m_hashes[hashPos];

            int storePos = m_nextfree;

            int currEntryPos = entryLocation;

            // Not overwriting
            while (currEntryPos > -1)
            {
                var entry = m_entries[currEntryPos];

                if (key == entry.Key)
                {
                    return currEntryPos;
                }
                currEntryPos = entry.Next;
            }

            m_hashes[hashPos] = storePos;

            m_entries[storePos].Next = entryLocation;
            m_entries[storePos].Key = key;
            m_entries[storePos].Value = value;
            m_entries[storePos].Hashcode = hash;

            ++m_nextfree;

            return storePos;
        }

        public int AddOverwrite(int key, int value)
        {
            if (m_nextfree >= m_entries.Length)
                Resize();

            uint hash = (uint)key;

            uint hashPos = hash % (uint)m_hashes.Length;

            int entryLocation = m_hashes[hashPos];

            int storePos = m_nextfree;

            int currEntryPos = entryLocation;

            // Overwriting
            while (currEntryPos > -1)
            {
                var entry = m_entries[currEntryPos];
                if (key == entry.Key)
                {
                    storePos = currEntryPos;
                    break; // do not increment nextfree - overwriting the value
                }
                currEntryPos = entry.Next;
            }

            ++m_nextfree;

            m_hashes[hashPos] = storePos;

            m_entries[storePos].Next = entryLocation;
            m_entries[storePos].Key = key;
            m_entries[storePos].Value = value;
            m_entries[storePos].Hashcode = hash;

            return storePos;
        }

        private void Resize()
        {
            uint newsize = FindNewSize();

            var newhashes = m_hashes;
            var newentries = m_entries;
            if (newsize > m_hashes.Length)
            {
                newhashes = new int[newsize];
                for (int i = 0; i < newhashes.Length; i++)
                {
                    newhashes[i] = -1;
                }
                newentries = new DictionaryEntry[newsize];
                Array.Copy(m_entries, newentries, m_nextfree);
            }

            for (int i = 0; i < m_nextfree; i++)
            {
                uint pos = newentries[i].Hashcode % newsize;
                int prevpos = newhashes[pos];
                newhashes[pos] = i;

                if (prevpos != -1)
                    newentries[i].Next = prevpos;
            }

            m_hashes = newhashes;
            m_entries = newentries;
        }

        private uint FindNewSize()
        {
            uint roughsize = (uint)m_hashes.Length * 2 + 1;

            for (int i = 0; i < m_primeSizes.Length; i++)
            {
                if (m_primeSizes[i] >= roughsize)
                    return m_primeSizes[i];
            }

            throw new NotImplementedException("Too large array");
        }

        public int Get(int key)
        {
            int pos = GetPosition(key);

            if (pos == -1)
                throw new Exception("Key does not exist");

            return m_entries[pos].Value;
        }

        public int GetPosition(int key)
        {
            uint hash = (uint)key.GetHashCode();

            uint pos = hash % (uint)m_hashes.Length;

            int entryLocation = m_hashes[pos];

            if (entryLocation == -1)
                return -1;

            int nextpos = entryLocation;

            do
            {
                DictionaryEntry entry = m_entries[nextpos];

                if (key.Equals(entry.Key))
                    return nextpos;

                nextpos = entry.Next;

            } while (nextpos != -1);

            return -1;
        }

        public bool ContainsKey(int key)
        {
            return GetPosition(key) != -1;
        }

        public ICollection<int> Keys
        {
            get { throw new NotImplementedException(); }
        }

        public bool Remove(int key)
        {
            throw new NotImplementedException();
        }

        public bool TryGetValue(int key, out int value)
        {
            int pos = GetPosition(key);

            if (pos == -1)
            {
                value = default(int);
                return false;
            }

            value = m_entries[pos].Value;

            return true;
        }

        public ICollection<int> Values
        {
            get { throw new NotImplementedException(); }
        }

        public int this[int key]
        {
            get
            {
                return Get(key);
            }
            set
            {
                AddOverwrite(key, value);
            }
        }

        public void Add(KeyValuePair<int, int> item)
        {
            int pos = AddNew(item.Key, item.Value);

            if (pos + 1 != m_nextfree)
                throw new Exception("Key already exists");
        }

        void IDictionary<int, int>.Add(int key, int value)
        {
            int pos = AddNew(key, value);

            if (pos + 1 != m_nextfree)
                throw new Exception("Key already exists");
        }

        public void Clear()
        {
            ClearHashes();
            m_nextfree = 0;
        }

        private void ClearHashes()
        {
            for (int i = 0; i < m_hashes.Length; i++)
            {
                m_hashes[i] = -1;
            }
        }

        public bool Contains(KeyValuePair<int, int> item)
        {
            int value;

            if (!TryGetValue(item.Key, out value))
                return false;

            if (!item.Value.Equals(value))
                return false;

            return true;
        }

        public void CopyTo(KeyValuePair<int, int>[] array, int arrayIndex)
        {
            throw new NotImplementedException();
        }

        public int Count
        {
            get { return m_nextfree; }
        }

        public bool IsReadOnly
        {
            get { return false; }
        }

        public bool Remove(KeyValuePair<int, int> item)
        {
            throw new NotImplementedException();
        }

        IEnumerator<KeyValuePair<int, int>> IEnumerable<KeyValuePair<int, int>>.GetEnumerator()
        {
            for (int i = 0; i < m_nextfree; i++)
            {
                yield return new KeyValuePair<int, int>(m_entries[i].Key, m_entries[i].Value);
            }
        }

        public Enumerator GetEnumerator()
        {
            return new Enumerator(m_entries, m_nextfree);
        }

        public struct Enumerator : IEnumerator<KeyValuePair<int, int>>
        {
            readonly DictionaryEntry[] m_entries;
            readonly int m_end;
            int m_current;

            internal Enumerator(DictionaryEntry[] entries, int end)
            {
                m_entries = entries;
                m_end = end;
                m_current = -1;
            }

            public KeyValuePair<int, int> Current
            {
                get { return new KeyValuePair<int, int>(m_entries[m_current].Key, m_entries[m_current].Value); }
            }

            object IEnumerator.Current
            {
                get { return new KeyValuePair<int, int>(m_entries[m_current].Key, m_entries[m_current].Value); }
            }

            public bool MoveNext()
            {
                ++m_current; // Do not care about overflow
                return m_current < m_end;
            }

            public void Reset()
            {
                m_current = -1;
            }

            public void Dispose()
            { }
        }


        global::System.Collections.IEnumerator global::System.Collections.IEnumerable.GetEnumerator()
        {
            for (int i = 0; i < m_nextfree; i++)
            {
                yield return new KeyValuePair<int, int>(m_entries[i].Key, m_entries[i].Value);
            }
        }

        public void Add(object key, object value)
        {
            int pos = AddNew((int)key, (int)value);

            if (pos + 1 != m_nextfree)
                throw new Exception("Key already exists");
        }

        public bool Contains(object key)
        {
            return Contains((int)key);
        }

        IDictionaryEnumerator IDictionary.GetEnumerator()
        {
            throw new NotImplementedException();
        }

        public bool IsFixedSize
        {
            get { throw new NotImplementedException(); }
        }

        ICollection IDictionary.Keys
        {
            get { throw new NotImplementedException(); }
        }

        public void Remove(object key)
        {
            throw new NotImplementedException();
        }

        ICollection IDictionary.Values
        {
            get { throw new NotImplementedException(); }
        }

        public object this[object key]
        {
            get
            {
                return this[(int)key];
            }
            set
            {
                this[(int)key] = (int)value;
            }
        }

        public void CopyTo(Array array, int index)
        {
            throw new NotImplementedException();
        }

        public bool IsSynchronized
        {
            get { throw new NotImplementedException(); }
        }

        public object SyncRoot
        {
            get { throw new NotImplementedException(); }
        }
    }
}
