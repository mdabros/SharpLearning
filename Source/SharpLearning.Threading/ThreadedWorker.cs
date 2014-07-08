using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace SharpLearning.Threading
{
    /// <summary>
    /// Simple threading class for multithreaded work 
    /// </summary>
    /// <typeparam name="TResult"></typeparam>
    public sealed class ThreadedWorker<TResult>
    {
        readonly int m_numberOfThreads;

        /// <summary>
        /// Takes the number of threads as argument
        /// </summary>
        /// <param name="numberOfTreads"></param>
        public ThreadedWorker(int numberOfTreads)
        {
            if (numberOfTreads < 1) { throw new ArgumentException("Number of threads must be at least 1"); }
            m_numberOfThreads = numberOfTreads;
        }

        /// <summary>
        /// Starts the threads and performs the work provided in tasks. 
        /// Each task has to add its results to the ConcurrentBag.
        /// </summary>
        /// <param name="tasks"></param>
        /// <param name="results"></param>
        public void Run(ConcurrentQueue<Action<ConcurrentBag<TResult>>> tasks, ConcurrentBag<TResult> results)
        {
            var threads = new List<Thread>();
            for (int i = 0; i < m_numberOfThreads; i++)
            {
                Thread t = new Thread(new ThreadStart(() => DoWork(tasks, results)));
                t.Start();
                threads.Add(t);
            }
            foreach (Thread t in threads) t.Join();
        }

        void DoWork(ConcurrentQueue<Action<ConcurrentBag<TResult>>> tasks, ConcurrentBag<TResult> results)
        {
            Action<ConcurrentBag<TResult>> task = null;
            while(tasks.TryDequeue(out task))
            {
                task(results);
            }
        }
    }
}
